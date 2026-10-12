# -*- coding: utf-8 -*-
"""
跑 ChatGarment 推論：照片 → 款式描述（文字 JSON）＋ 76 個連續數值。

原作者的腳本用 deepspeed、flash-attention、bf16 全部放進顯示卡（約 16 GB），這裡改成依顯示卡自動選：
  --mode auto（預設）：顯示卡 ≥ 20 GB 用 bf16 原版，否則用 4-bit
  --mode 4bit：語言模型的線性層量化成 NF4，全部放進顯示卡（約 6～7 GB，8 GB 顯示卡可用）
  --mode bf16：原版精度。顯示卡夠大就整個放進去；放不下時分流到系統記憶體（很慢，要加 --confirm-bf16）

權重檔是「LoRA 尚未合併」的完整狀態（含 CLIP），載入時用記憶體映射讀取，
在 CPU 上把 LoRA 合併進主體，所以不需要另外下載 LLaVA-1.5-7B 和 CLIP 的權重，
只會從 Hugging Face 抓它們的設定檔與分詞器（幾 MB）。

要用 .venv-chatgarment 的 Python（新版 PyTorch，支援 RTX 50 系列）：
    .venv-chatgarment\\Scripts\\python tools\\chatgarment\\infer.py --mode 4bit --images <資料夾> --out <資料夾>

每張照片在輸出資料夾產生 <名稱>/：output.txt（模型輸出的文字）、output.json（修復後的 JSON）、
float_preds.npy（76 個數值）、meta.json（耗時、顯示記憶體用量）。
轉成 design.yaml 要另外跑 to_design.py（用 GarmentCode 的 .venv）。
"""
import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
AI_VENV = REPO / ".venv-chatgarment"
if Path(sys.prefix).resolve() != AI_VENV.resolve() and (AI_VENV / "Scripts" / "python.exe").exists():
    # 用錯環境時（例如在 VS Code 按 ▶，預設是 .venv）自動換成 ChatGarment 環境重新執行
    import subprocess
    sys.exit(subprocess.call([str(AI_VENV / "Scripts" / "python.exe"), __file__, *sys.argv[1:]]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

CG_ROOT = REPO / "third_party" / "ChatGarment"
CKPT = CG_ROOT / "checkpoints" / "try_7b_lr1e_4_v3_garmentcontrol_4h100_v4_final" / "pytorch_model.bin"
BASE = "liuhaotian/llava-v1.5-7b"
CLIP = "openai/clip-vit-large-patch14-336"
LORA_SCALE = 256 / 128  # 作者訓練時 lora_alpha / lora_r

sys.path.insert(0, str(CG_ROOT))
from llava import conversation as conversation_lib  # noqa: E402
from llava.constants import DEFAULT_IMAGE_TOKEN  # noqa: E402
from llava.json_fixer import repair_json  # noqa: E402
from llava.mm_utils import tokenizer_image_token  # noqa: E402
from llava.model.language_model.llava_garment_float50 import (  # noqa: E402
    GarmentGPTFloat50ForCausalLM, LlavaConfig)

QUESTIONS = [
    "Can you describe the geometry features of the garments worn by the model in the Json format?",
    "Can you estimate the sewing pattern code based on the image and Json format garment geometry description?",
]


def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE, use_fast=False, model_max_length=3072, padding_side="right")
    tok.pad_token = tok.unk_token
    tok.add_tokens("[SEG]")
    seg = tok("[SEG]", add_special_tokens=False).input_ids[-1]
    return tok, seg


def merged_state_dict(path):
    """讀 PEFT 格式的權重，把 LoRA（只加在 q_proj、v_proj）合併進主體，key 改回一般模型的名稱。"""
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
    out, n_merged = {}, 0
    for key, value in sd.items():
        if ".lora_A." in key or ".lora_B." in key:
            continue
        name = key.removeprefix("base_model.model.")
        if name.endswith(".base_layer.weight"):
            prefix = key[: -len(".base_layer.weight")]
            a = sd[f"{prefix}.lora_A.default.weight"].float()
            b = sd[f"{prefix}.lora_B.default.weight"].float()
            value = (value.float() + LORA_SCALE * (b @ a)).to(torch.bfloat16)
            name = name.replace(".base_layer.weight", ".weight")
            n_merged += 1
        elif name.endswith(".base_layer.bias"):
            name = name.replace(".base_layer.bias", ".bias")
        out[name] = value
    return out, n_merged


def build_model(seg_token_idx, vocab_size):
    from accelerate import init_empty_weights
    from transformers import CLIPImageProcessor, CLIPVisionModel

    config = LlavaConfig.from_pretrained(BASE)
    config.vocab_size = vocab_size
    config.use_cache = True
    config._attn_implementation = "sdpa"
    with init_empty_weights():
        model = GarmentGPTFloat50ForCausalLM(config, seg_token_idx=seg_token_idx)
        tower = model.get_model().get_vision_tower()
        tower.vision_tower = CLIPVisionModel(tower.cfg_only)  # 權重從 ChatGarment 的檔案載入
    tower.image_processor = CLIPImageProcessor.from_pretrained(CLIP)
    tower.is_loaded = True
    return model


def quantize_to_gpu(model):
    """語言模型 32 層的 7 個線性層換成 bitsandbytes 的 NF4；其餘（embedding、lm_head、
    float_layer、CLIP、projector）維持 bf16。逐層處理，避免一次佔用太多記憶體。"""
    import bitsandbytes as bnb
    names = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
    for layer in model.model.layers:
        for name in names:
            parent_name, child = name.split(".")
            parent = getattr(layer, parent_name)
            old = getattr(parent, child)
            new = bnb.nn.Linear4bit(old.in_features, old.out_features, bias=False,
                                    compute_dtype=torch.bfloat16, quant_type="nf4", device="meta")
            new.weight = bnb.nn.Params4bit(old.weight.data.contiguous(), requires_grad=False,
                                           quant_type="nf4", compress_statistics=True)
            setattr(parent, child, new.cuda())
        layer.to("cuda")
    model.to("cuda")
    return model


def offload_bf16(model, gpu_gib):
    from accelerate import dispatch_model, infer_auto_device_map
    device_map = infer_auto_device_map(
        model, max_memory={0: f"{gpu_gib}GiB", "cpu": "26GiB"},
        no_split_module_classes=["LlamaDecoderLayer", "CLIPEncoderLayer"], dtype=torch.bfloat16)
    # 前處理和最後的輸出層固定放顯示卡
    for key in list(device_map):
        if any(s in key for s in ["vision_tower", "mm_projector", "embed_tokens", "lm_head", "float_layer", "model.norm"]):
            device_map[key] = 0
    return dispatch_model(model, device_map=device_map), device_map


def preprocess(image_path, processor):
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    if w != h:  # 補成正方形，和作者的 image_aspect_ratio='pad' 相同
        side = max(w, h)
        bg = Image.new("RGB", (side, side), tuple(int(x * 255) for x in processor.image_mean))
        bg.paste(img, ((side - w) // 2, (side - h) // 2))
        img = bg
    return processor.preprocess(img, return_tensors="pt")["pixel_values"][0]


def run_one(model, tok, image, max_new_tokens):
    """作者的兩步驟：先請模型用 JSON 描述衣服，再根據描述輸出版型參數（文字＋[SEG] 位置的數值）。"""
    image = image.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
    text_output, float_preds = "", None
    for k, question in enumerate(QUESTIONS):
        conv = conversation_lib.conv_templates["v1"].copy()
        conv.messages = []
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        if k == 1:
            prompt += "\n" + text_output.replace("upper_garment", "upperbody_garment").replace(
                "lower_garment", "lowerbody_garment")
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(conv.get_prompt(), tok, return_tensors="pt").unsqueeze(0).to("cuda")
        output_ids, float_preds, _ = model.evaluate(image, image, input_ids, max_new_tokens=max_new_tokens, tokenizer=tok)
        text_output = tok.decode(output_ids[0, 1:], skip_special_tokens=False).strip().replace("</s>", "")
        if k == 0:
            first = text_output
    text_output = text_output.replace("[STARTS]", "").replace("[SEG]", "").replace("[ENDS]", "")
    return first, text_output, None if float_preds is None else float_preds.float().cpu().numpy()


BF16_FULL_GPU_GIB = 20  # 顯示卡記憶體至少這麼多，bf16 原版才能整個放進顯示卡（模型約 14 GB＋推論暫存）


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "4bit", "bf16"], default="auto",
                    help=f"auto：顯示卡 ≥ {BF16_FULL_GPU_GIB} GB 用 bf16 原版，否則用 4-bit")
    ap.add_argument("--images", default=str(REPO / "output" / "chatgarment_ai" / "inputs"),
                    help="照片資料夾，預設 output/chatgarment_ai/inputs")
    ap.add_argument("--out", default=None, help="輸出資料夾，預設 output/chatgarment_ai/<mode>")
    ap.add_argument("--gpu-gib", type=float, default=5.0, help="bf16 放不下時，分流模式放在顯示卡的上限")
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--confirm-bf16", action="store_true",
                    help="顯示卡放不下 bf16 時要分流到系統記憶體，會跑好幾個小時，要明確加這個參數才會執行")
    args = ap.parse_args()

    total_gib = torch.cuda.get_device_properties(0).total_memory / 2**30
    full_gpu = total_gib >= BF16_FULL_GPU_GIB
    if args.mode == "auto":
        args.mode = "bf16" if full_gpu else "4bit"
    print(f"顯示卡：{torch.cuda.get_device_name(0)}（{total_gib:.1f} GB）→ 使用 {args.mode}"
          + ("（原版精度，整個放進顯示卡）" if args.mode == "bf16" and full_gpu else ""))
    args.out = args.out or str(REPO / "output" / "chatgarment_ai" / args.mode)
    if args.mode == "bf16" and not full_gpu and not args.confirm_bf16:
        print(f"顯示卡不到 {BF16_FULL_GPU_GIB} GB，bf16 要分流到系統記憶體（每張約 1 小時），"
              "要加 --confirm-bf16 才會執行，這次跳過")
        return

    t0 = time.time()
    tok, seg = load_tokenizer()
    model = build_model(seg, len(tok))
    sd, n_merged = merged_state_dict(CKPT)
    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    print(f"合併 LoRA {n_merged} 個；缺少 {len(missing)} 個、多出 {len(unexpected)} 個權重")
    if missing or unexpected:
        print("  缺少：", missing[:5], "\n  多出：", unexpected[:5])
        raise SystemExit("權重名稱對不上，停止")
    model = model.to(torch.bfloat16).eval()

    if args.mode == "4bit":
        model = quantize_to_gpu(model)
    elif full_gpu:
        model = model.to("cuda")  # 原版精度，全部放顯示卡
    else:
        model, device_map = offload_bf16(model, args.gpu_gib)
        n_cpu = sum(1 for v in device_map.values() if v == "cpu")
        print(f"bf16 分流：{n_cpu} 個模組放系統記憶體")
    load_s = time.time() - t0
    print(f"載入完成 {load_s:.0f} 秒，顯示記憶體 {torch.cuda.memory_allocated() / 2**30:.2f} GiB")

    processor = model.get_model().get_vision_tower().image_processor
    out_root = Path(args.out)
    images = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    for path in images:
        dst = out_root / path.stem
        if (dst / "meta.json").exists():
            print(f"[略過] {path.name}（已經有結果）")
            continue
        dst.mkdir(parents=True, exist_ok=True)
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        first, text, floats = run_one(model, tok, preprocess(path, processor), args.max_new_tokens)
        secs = time.time() - t
        (dst / "output.txt").write_text(f"[描述]\n{first}\n\n[版型]\n{text}\n", encoding="utf-8")
        (dst / "output.json").write_text(
            json.dumps(repair_json(text, return_objects=True), ensure_ascii=False, indent=2), encoding="utf-8")
        if floats is not None:
            np.save(dst / "float_preds.npy", floats)
        meta = {"image": str(path), "mode": args.mode, "seconds": round(secs, 1),
                "peak_gpu_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "n_floats": None if floats is None else int(floats.size)}
        (dst / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[{path.name}] {secs:.0f} 秒，顯示記憶體峰值 {meta['peak_gpu_gib']} GiB，數值 {meta['n_floats']} 個")


if __name__ == "__main__":
    main()
