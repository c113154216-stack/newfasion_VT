#!/bin/bash
# OmniFabric 在 24 GB 顯示卡（RTX 3090）上跑：一件衣服的正面照（＋背面照）→ 版片＋UV 貼圖＋縫好的 3D 衣服。
#
# 改自 OmniFabric 的 scripts/run_pipeline.sh（Apache-2.0，commit 205c179），差別：
#   - FLUX.2 兩個步驟改用 lowvram/flux2_lowvram.py（文字編碼器和主模型分兩段載入）
#   - 第 6 步改用 lowvram/flux1_lowvram.py（FLUX.1 主模型分層搬運）
#   - 正面上色固定用 FLUX.2（全自動，不走 Gemini 手動流程）；多視角固定用 flux2，不需要 ComfyUI／MiniMax-H3
#   - 有背面照時，背面用真的照片上色（原版只吃一張照片，背面是 AI 生成的）
#
# 用法（在 OmniFabric 根目錄、conda activate omnifabric 之後）：
#   bash lowvram/run_3090.sh <正面照> [背面照] [輸出資料夾]
# 環境變數同原版：GPU、CHATGARMENT_CKPT、OMNIFABRIC_CKPT、SEED、GARMENTS（預設 upper：只做上衣）
#
# 輸出（每件預測出的衣服，例如 valid_garment_upper）：
#   <輸出>/<衣服>/normalized_texture.png   最終 UV 貼圖（版片空間）  ← 要帶回來的
#   <輸出>/<衣服>/editable_template.png    版片 UV 範本（哪一塊是哪片版片）
#   <輸出>/<衣服>/final/                   貼好貼圖、縫好的 3D 衣服（<衣服>_sim.obj ＋ 渲染圖）
#   <輸出>/<衣服>/views/                   正面上色結果 front_garment.png、背面 back_garment.png
# 重跑同一個輸出資料夾會沿用已經做好的版片、模擬與正背面圖（刪掉就重做）。

set -e

FRONT=$(realpath "$1")
BACK=""
[[ -n "$2" && -f "$2" ]] && BACK=$(realpath "$2")
NAME=$(basename "${FRONT%.*}")
if [[ -n "$BACK" ]]; then OUT_ARG="$3"; else OUT_ARG="$2"; fi
OUT=$(realpath -m "${OUT_ARG:-outputs/$NAME}")

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"
export GARMENTCODE_ROOT="${GARMENTCODE_ROOT:-$REPO_ROOT/third_party/GarmentCodeRC}"
export EGL_DEVICE_ID=${GPU:-0}
export PYTHONWARNINGS=ignore TRANSFORMERS_VERBOSITY=error DIFFUSERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True   # 減少顯示記憶體碎片，24 GB 比較不容易爆

GPU=${GPU:-0}
export CHATGARMENT_CKPT=$(realpath -m "${CHATGARMENT_CKPT:-checkpoints/chatgarment/pytorch_model.bin}")
OMNIFABRIC_CKPT=$(realpath -m "${OMNIFABRIC_CKPT:-checkpoints/omnifabric}")
SEED=${SEED:-42}
GARMENTS=${GARMENTS:-upper}

run() { CUDA_VISIBLE_DEVICES=$GPU python -m "$@"; }
t0=$SECONDS
step() { echo ""; echo "=== $1（已經過 $(( (SECONDS - t0) / 60 )) 分） ==="; }

python -m omnifabric.utils.check_checkpoints --output_dir "$OUT" \
    --texture_backend flux2 --views_backend flux2 \
    --chatgarment_ckpt "$CHATGARMENT_CKPT" --omnifabric_ckpt "$OMNIFABRIC_CKPT" --comfy_server http://127.0.0.1:8188

mkdir -p "$OUT/input"
[[ "$FRONT" -ef "$OUT/input/$(basename "$FRONT")" ]] || cp "$FRONT" "$OUT/input/"
[[ -n "$BACK" ]] && cp "$BACK" "$OUT/back_photo.${BACK##*.}"


# ===============================
step "[1/6] ChatGarment 預測版片並模擬"
export CHATGARMENT_OUTPUT_DIR="$OUT/chatgarment"
if [[ ! -f "$CHATGARMENT_OUTPUT_DIR/vis_new/all_json_spec_files.json" ]]; then
    (cd third_party/ChatGarment && CUDA_VISIBLE_DEVICES=$GPU python "$REPO_ROOT/omnifabric/sewing/predict_pattern.py" \
        --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
        --model_name_or_path liuhaotian/llava-v1.5-7b \
        --version v1 \
        --data_path ./ \
        --data_path_eval "$OUT/input" \
        --image_folder ./ \
        --vision_tower openai/clip-vit-large-patch14-336 \
        --mm_projector_type mlp2x_gelu \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio pad \
        --group_by_modality_length True \
        --bf16 True \
        --output_dir "$OUT/chatgarment/llava_args" \
        --per_device_eval_batch_size 1 \
        --model_max_length 3072 \
        --gradient_checkpointing True \
        --lazy_preprocess True \
        --report_to none)
    run omnifabric.sewing.simulate --all_paths_json "$CHATGARMENT_OUTPUT_DIR"
fi

GARMENT_SPECS=$(python -m omnifabric.sewing.select_garments "$CHATGARMENT_OUTPUT_DIR/vis_new/all_json_spec_files.json" \
    --garments "$GARMENTS")


for GARMENT_SPEC in $GARMENT_SPECS; do
    GARMENT_SPEC=$(realpath "$GARMENT_SPEC")
    GARMENT_NAME=$(basename "$GARMENT_SPEC" | sed 's/_specification\.json$//')
    G="$OUT/$GARMENT_NAME"
    SIM="$G/sim/$GARMENT_NAME"
    VIEWS="$G/views"
    mkdir -p "$G" "$VIEWS"
    echo ""
    echo "################ $GARMENT_NAME ################"
    render_only() { echo "$G/sim/${GARMENT_NAME}_render_garment_only_$1.png"; }


    # ===============================
    step "[2/6] UV 範本、位置圖、灰色模擬"
    if [[ ! -f "$SIM/${GARMENT_NAME}_sim.obj" ]]; then
        python -m omnifabric.sewing.uv_template "$GARMENT_SPEC" --output_dir "$G" --template_name editable_template.png
        python -m omnifabric.sewing.position_map "$GARMENT_SPEC" --output_dir "$G"
        run omnifabric.sewing.textured_sim "$GARMENT_SPEC" assets/lightgrey.png --output_dir "$G/sim_tmp"
        rm -rf "$G/sim" && mv "$G/sim_tmp/${GARMENT_NAME}_textured" "$G/sim" && rmdir "$G/sim_tmp"
    fi


    # ===============================
    step "[3/6] 正面：用正面照替灰色衣服上色（FLUX.2，分兩段載入）"
    python -m omnifabric.utils.crop_ref --input "$SIM/${GARMENT_NAME}_render_front.png" \
        --output "$SIM/${GARMENT_NAME}_render_front_cropped.png" --ratio 0.85 --vertical_offset -140
    python -m omnifabric.utils.color_background "$SIM/${GARMENT_NAME}_render_front_cropped.png" \
        "$SIM/${GARMENT_NAME}_render_front_cropped_chroma_key.png"
    python -m omnifabric.utils.color_background "$(render_only front)" "$VIEWS/render_front_chroma_key.png"

    [[ -f "$VIEWS/front_garment.png" ]] || run lowvram.flux2_lowvram texture_transfer \
        --task texture-transfer \
        --image1 "$VIEWS/render_front_chroma_key.png" \
        --image2 "$FRONT" \
        -o "$VIEWS/front_garment.png" \
        --seed "$SEED" --guidance 5.0 --steps 25

    python -m omnifabric.utils.composite_garment_texture \
        --mannequin "$SIM/${GARMENT_NAME}_render_front_cropped_chroma_key.png" \
        --mask "$SIM/${GARMENT_NAME}_render_front_cropped_mask.png" \
        --reference "$VIEWS/front_garment.png" \
        --output "$VIEWS/front_human.png"
    python -m omnifabric.utils.remove_background "$SIM/${GARMENT_NAME}_render_front_cropped.png" \
        "$VIEWS/front_human.png" "$G/frontal_rendering.png"


    # ===============================
    PROJ_IN="$G/projection/inputs/$GARMENT_NAME"
    mkdir -p "$PROJ_IN"
    python -m omnifabric.utils.fit_view "$(render_only front)" "$VIEWS/front_garment.png" "$PROJ_IN/front_view.png" --aligned

    if [[ -n "$BACK" ]]; then
        step "[4/6] 背面：用真的背面照上色（FLUX.2），投影到網格"
        python -m omnifabric.utils.color_background "$(render_only back)" "$VIEWS/render_back_chroma_key.png"
        [[ -f "$VIEWS/back_garment.png" ]] || run lowvram.flux2_lowvram texture_transfer --back \
            --task texture-transfer \
            --image1 "$VIEWS/render_back_chroma_key.png" \
            --image2 "$BACK" \
            -o "$VIEWS/back_garment.png" \
            --seed "$SEED" --guidance 5.0 --steps 25
    else
        step "[4/6] 背面：沒有背面照，由 FLUX.2 從正面推測，投影到網格"
        [[ -f "$VIEWS/back_garment.png" ]] || run lowvram.flux2_lowvram back_view_flux2 --image "$VIEWS/front_garment.png" \
            -o "$VIEWS/back_garment.png" --width 1072 --height 1072 --seed 469
    fi
    python -m omnifabric.utils.remove_background "$(render_only back)" "$VIEWS/back_garment.png" "$PROJ_IN/back_view.png"

    cp "$SIM/${GARMENT_NAME}_sim.obj" "$PROJ_IN/mesh.obj"
    run omnifabric.coarse.project_multiview --input_root "$G/projection/inputs" \
        --output_root "$G/projection/outputs" --example_name "$GARMENT_NAME"


    # ===============================
    step "[5/6] 投影結果烘到版片 UV（粗貼圖）"
    python -m omnifabric.coarse.extract_uv_texture \
        --obj_path "$PROJ_IN/mesh.obj" \
        --glb_path "$G/projection/outputs/results_multiview_general_${GARMENT_NAME}_full/mesh_multiview_full_${GARMENT_NAME}_textured_mesh.glb" \
        --template_path "$G/editable_template.png" \
        --output_path "$G/coarse_texture.png"


    # ===============================
    step "[6/6] 整理貼圖（FLUX.1＋OmniFabric LoRA，分層搬運，最慢）並重新模擬"
    python -m omnifabric.normalize.square_layout to_square --template "$G/editable_template.png" \
        --coarse "$G/coarse_texture.png" --pos "$G/${GARMENT_NAME}_position_map.pt" --out_dir "$G/normalize/inputs"
    NORMALIZED_PARTS=()
    for PART_DIR in "$G"/normalize/inputs/part*; do
        PART=$(basename "$PART_DIR")
        run lowvram.flux1_lowvram \
            --checkpoint "$OMNIFABRIC_CKPT" \
            --mask_path "$PART_DIR/editable_template.png" \
            --proj_mask_path "$PART_DIR/coarse_texture.png" \
            --pos_path "$PART_DIR/position_map.pt" \
            --ref_path "$G/frontal_rendering.png" \
            --prompt "A textured sewing pattern" \
            --output "$G/normalize/$PART/normalized.png" \
            --use_proj_mask \
            --target_size 1024 \
            --ref_size 512 \
            --save_conditions \
            --postprocess \
            --use_sr \
            --seed "$SEED" \
            --denoise_full
        NORMALIZED_PARTS+=("$G/normalize/$PART/normalized_fullres.png")
    done

    python -m omnifabric.normalize.square_layout from_square "${NORMALIZED_PARTS[@]}" \
        --layout "$G/normalize/inputs/layout.json" --output "$G/normalize/normalized_fullres_restored.png"
    python -m omnifabric.normalize.dilate_uv_panels "$G/normalize/normalized_fullres_restored.png" "$G/editable_template.png" \
        --N 10 --heal_radius 1 --output "$G/normalized_texture.png"

    run omnifabric.sewing.textured_sim "$GARMENT_SPEC" "$G/normalized_texture.png" --output_dir "$G/final_tmp"
    rm -rf "$G/final" && mv "$G/final_tmp/${GARMENT_NAME}_textured" "$G/final" && rmdir "$G/final_tmp"

    echo ""
    echo "完成：$G/normalized_texture.png（共 $(( (SECONDS - t0) / 60 )) 分）"
done
