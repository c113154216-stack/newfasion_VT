#!/bin/bash
# 一次跑一整個資料夾的衣服，跑完把要帶回來的檔案收進 <輸出>/collect/。
#
#   bash lowvram/batch_3090.sh <照片資料夾> [輸出資料夾，預設 outputs/batch] [只跑名稱開頭是這個的，例如 h05]
#
# 照片命名：<衣服>_front.jpg（必要）、<衣服>_back.jpg（可以沒有，沒有就由 AI 推測背面）；png、jpg、jpeg、webp 都可以。
# 一件失敗不會中斷整批，最後列出失敗的衣服；log 在 <輸出>/<衣服>.log。
# 帶回來的 collect/<衣服>/：
#   normalized_texture.png   最終 UV 貼圖
#   editable_template.png    版片 UV 範本
#   front_garment.png、back_garment.png   正、背面上色結果（檢查用）
#   final/                   貼好貼圖、縫好的 3D 衣服（obj、mtl、渲染圖）
#   design/                  ChatGarment 預測的款式（specification.json、design.yaml）

PHOTOS=$(realpath "$1")
OUT=$(realpath -m "${2:-outputs/batch}")
cd "$(dirname "$0")/.."
mkdir -p "$OUT/collect"
failed=()

for FRONT in "$PHOTOS"/${3}*_front.*; do
    [[ -f "$FRONT" ]] || continue
    NAME=$(basename "$FRONT"); NAME=${NAME%_front.*}
    BACK=$(ls "$PHOTOS/${NAME}_back."* 2>/dev/null | head -n 1)
    echo "######## $NAME（背面照：${BACK:-沒有}）"
    if bash lowvram/run_3090.sh "$FRONT" $BACK "$OUT/$NAME" > "$OUT/$NAME.log" 2>&1; then
        for G in "$OUT/$NAME"/valid_garment_*/; do
            [[ -f "$G/normalized_texture.png" ]] || continue
            C="$OUT/collect/$NAME/$(basename "$G")"
            mkdir -p "$C/design"
            cp "$G/normalized_texture.png" "$G/editable_template.png" "$C/"
            cp "$G/views/front_garment.png" "$G/views/back_garment.png" "$C/" 2>/dev/null
            cp -r "$G/final" "$C/"
            find "$OUT/$NAME/chatgarment" -name "*$(basename "$G")*specification.json" -exec cp {} "$C/design/" \; 2>/dev/null
            find "$OUT/$NAME/chatgarment" -name "design*.yaml" -exec cp {} "$C/design/" \; 2>/dev/null
        done
        echo "  完成，$(tail -n 1 "$OUT/$NAME.log")"
    else
        echo "  失敗，看 $OUT/$NAME.log 最後幾行"
        failed+=("$NAME")
    fi
done

echo ""
echo "要帶回來的檔案：$OUT/collect/"
[[ ${#failed[@]} -gt 0 ]] && echo "失敗：${failed[*]}"
exit 0
