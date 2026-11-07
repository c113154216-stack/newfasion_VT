# -*- coding: utf-8 -*-
"""
OmniFabric 第 6 步（FLUX.1-dev ＋ OmniFabric LoRA 整理貼圖）改成 24 GB 顯示卡也跑得動。

原版整個 pipeline（bf16）放進顯示卡約 35 GB：主模型 12B 參數就有約 24 GB，T5 文字編碼器約 9.5 GB。
這裡改成：
  - 主模型（transformer）的權重放系統記憶體，用 diffusers 的 group offload（leaf_level、use_stream）
    算到哪一層才搬那一層上顯示卡，算完搬回去。精度不變（仍是 bf16、同一組 LoRA），只是比較慢
  - T5、CLIP、VAE 照常放顯示卡（合計約 10 GB）
做法是換掉原程式的 load_pipeline()，原本的程式碼一行都不改。

需要系統記憶體至少約 40 GB（主模型 24 GB 要鎖在記憶體裡傳輸）。系統記憶體不夠時設 LOWVRAM_NO_STREAM=1：
不鎖記憶體、不預先搬下一層，比較省記憶體但更慢。

在 OmniFabric 根目錄執行（參數和 omnifabric.normalize.infer 完全相同）：
    python -m lowvram.flux1_lowvram --checkpoint checkpoints/omnifabric --mask_path … （其餘同原版）
"""
import os
import sys
from pathlib import Path

import torch


def load_pipeline_lowvram(checkpoint_path, base_model="black-forest-labs/FLUX.1-dev", device="cuda",
                          dtype=torch.bfloat16, cache_dir=None):
    from diffusers.pipelines import FluxPipeline
    kwargs = {"torch_dtype": dtype}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    pipe = FluxPipeline.from_pretrained(base_model, **kwargs)  # 先全部放 CPU
    path = Path(checkpoint_path)
    if not (path / "default.safetensors").exists():
        raise FileNotFoundError(f"找不到 OmniFabric LoRA：{path / 'default.safetensors'}（見 README 的下載步驟）")
    # LoRA 要在 offload 之前載入，LoRA 的小矩陣也會一起被搬上搬下
    pipe.load_lora_weights(str(path), weight_name="default.safetensors", adapter_name="default")
    for name in ("text_encoder", "text_encoder_2", "vae"):
        getattr(pipe, name).to(device)
    stream = os.environ.get("LOWVRAM_NO_STREAM") != "1"
    pipe.transformer.enable_group_offload(onload_device=torch.device(device), offload_device=torch.device("cpu"),
                                          offload_type="leaf_level", use_stream=stream)
    # 主模型的權重平常在 CPU，pipeline 會以為要在 CPU 上算；OmniFabric 的 generate() 用這兩個屬性放張量
    cls = type(pipe)
    cls._execution_device = property(lambda self: torch.device(device))
    cls.device = property(lambda self: torch.device(device))
    print(f"[lowvram] FLUX.1 主模型分層搬運，顯示記憶體 {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    return pipe


def main():
    import omnifabric.normalize.infer as infer
    infer.load_pipeline = load_pipeline_lowvram
    sys.argv[0] = "omnifabric.normalize.infer"
    infer.main()


if __name__ == "__main__":
    main()
