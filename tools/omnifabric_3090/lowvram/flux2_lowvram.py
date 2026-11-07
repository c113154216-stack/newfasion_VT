# -*- coding: utf-8 -*-
"""
OmniFabric 的 FLUX.2-dev 步驟（正面上色 texture_transfer、背面生成 back_view_flux2）改成 24 GB 顯示卡也跑得動。

原版把 Mistral 3 文字編碼器（4-bit 約 13 GB）和 FLUX.2 主模型（4-bit 約 17 GB）同時放進顯示卡，共約 28 GB。
這裡改成兩段：
  1. 只載入文字編碼器：改寫提示詞（upsample_prompt）、算出 prompt_embeds，然後整個釋放
  2. 再載入 FLUX.2 主模型，用算好的 prompt_embeds 生圖
做法是換掉原程式的 load_pipeline()，原本的程式碼一行都不改，輸出應該和原版相同（同樣的 4-bit 模型、同樣的種子），
只是多花一次載入模型的時間。

在 OmniFabric 根目錄執行（參數和原本的模組完全相同）：
    python -m lowvram.flux2_lowvram texture_transfer --task texture-transfer --image1 … --image2 … -o …
    python -m lowvram.flux2_lowvram texture_transfer --back --task texture-transfer --image1 背面灰模 --image2 背面照 -o …
    python -m lowvram.flux2_lowvram back_view_flux2 --image … -o …
--back：用真的背面照替背面灰模上色（OmniFabric 原本只有正面照，背面是 AI 生成的），提示詞裡的 frontal 換成 back view。
第 2 段生圖時顯示記憶體還是不夠，可以設 FLUX2_MAX_SIDE=768（輸出長邊上限，像素），解析度降低換記憶體。
"""
import gc
import importlib
import os
import sys

import torch


class TwoStageFlux2:
    """看起來像 Flux2Pipeline：改寫提示詞時用文字編碼器，呼叫生圖時才換成主模型。"""

    def __init__(self, tt):
        self.tt = tt  # omnifabric.coarse.texture_transfer 模組（REPO_ID、HF_CACHE、NF4_KWARGS）
        text_encoder = tt.Mistral3ForConditionalGeneration.from_pretrained(
            tt.REPO_ID, subfolder="text_encoder", quantization_config=tt.TransformersNF4Config(**tt.NF4_KWARGS),
            torch_dtype=torch.bfloat16, cache_dir=tt.HF_CACHE)
        tokenizer = tt.PixtralProcessor.from_pretrained(tt.REPO_ID, subfolder="tokenizer", cache_dir=tt.HF_CACHE)
        self.text_pipe = tt.Flux2Pipeline.from_pretrained(
            tt.REPO_ID, text_encoder=text_encoder, tokenizer=tokenizer, transformer=None, vae=None,
            torch_dtype=torch.bfloat16, cache_dir=tt.HF_CACHE)
        print(f"[lowvram] 第 1 段：文字編碼器，顯示記憶體 {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    def __getattr__(self, name):  # upsample_prompt、system_message 等都交給文字編碼器那一段
        return getattr(self.text_pipe, name)

    def __call__(self, prompt=None, **kwargs):
        tt = self.tt
        with torch.no_grad():
            embeds, _ = self.text_pipe.encode_prompt(prompt=prompt, device="cuda",
                                                     max_sequence_length=kwargs.get("max_sequence_length", 512))
        embeds = embeds.cpu()
        tokenizer, scheduler = self.text_pipe.tokenizer, self.text_pipe.scheduler
        del self.text_pipe
        gc.collect()
        torch.cuda.empty_cache()
        transformer = tt.Flux2Transformer2DModel.from_pretrained(
            tt.REPO_ID, subfolder="transformer", quantization_config=tt.DiffusersNF4Config(**tt.NF4_KWARGS),
            torch_dtype=torch.bfloat16, cache_dir=tt.HF_CACHE)
        pipe = tt.Flux2Pipeline.from_pretrained(
            tt.REPO_ID, text_encoder=None, tokenizer=tokenizer, transformer=transformer, scheduler=scheduler,
            torch_dtype=torch.bfloat16, cache_dir=tt.HF_CACHE)
        pipe.vae = pipe.vae.to("cuda", dtype=torch.bfloat16)  # 4-bit 的主模型不能搬，只搬 VAE（和原版相同）
        print(f"[lowvram] 第 2 段：FLUX.2 主模型，顯示記憶體 {torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
        max_side = int(os.environ.get("FLUX2_MAX_SIDE", "0"))
        w, h = kwargs.get("width"), kwargs.get("height")
        if max_side and w and h and max(w, h) > max_side:
            scale = max_side / max(w, h)
            kwargs["width"], kwargs["height"] = int(w * scale) // 16 * 16, int(h * scale) // 16 * 16
            print(f"[lowvram] 輸出 {w}×{h} → {kwargs['width']}×{kwargs['height']}（FLUX2_MAX_SIDE）", flush=True)
        return pipe(prompt=None, prompt_embeds=embeds.to("cuda"), **kwargs)


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("texture_transfer", "back_view_flux2"):
        raise SystemExit("用法：python -m lowvram.flux2_lowvram texture_transfer|back_view_flux2 [原本的參數…]")
    name = sys.argv.pop(1)
    back = "--back" in sys.argv
    if back:
        sys.argv.remove("--back")
    tt = importlib.import_module("omnifabric.coarse.texture_transfer")
    target = importlib.import_module(f"omnifabric.coarse.{name}")
    target.load_pipeline = lambda: TwoStageFlux2(tt)  # back_view_flux2 是 from … import load_pipeline，要換它自己的名字
    if back:
        tt.TASK_PROMPTS["texture-transfer"] = (tt.TASK_PROMPTS["texture-transfer"]
                                               .replace("frontal product photo", "back-view product photo")
                                               .replace("frontal", "back view"))
    sys.argv[0] = f"omnifabric.coarse.{name}"
    target.main()


if __name__ == "__main__":
    main()
