# -*- coding: utf-8 -*-
"""
和 ChatGarment 互動的網站：照片 → 衣服、文字 → 衣服、修改衣服、自由提問，結果可以直接看版片和 3D。

模型只載入一次，留在顯示卡上，網頁每按一次就問一次。問法照作者的評估腳本
（scripts/evaluate_garment_v2_*_1float.py），差別是作者用 GPT-4o 把自然語言轉成結構化描述，
這裡改成在網頁上直接填結構化描述（免費、不用 API）。

要用 .venv-chatgarment 的 Python（有顯示卡的電腦）：
    .venv-chatgarment\\Scripts\\python flows\\body_fit\\chat_server.py [--mode auto|4bit|bf16] [--port 8800]
沒有顯示卡時可以用 --mock 測網頁：不載入模型，回傳 results/chatgarment_ai 裡現成的結果。
瀏覽器開 http://127.0.0.1:8800/

版片和 3D 由 chat_worker.py 產生，用 GarmentCode 的 .venv（環境變數 GC_PYTHON 可以改；路徑不能有中文）。
每次問答存在 output/chatgarment_chat/<編號>/，格式和 infer.py 的輸出相同（output.txt、output.json、float_preds.npy）。
"""
import argparse
import base64
import io
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CG_ROOT = REPO / "third_party" / "ChatGarment"
GC_ROOT = REPO / "third_party" / "GarmentCodeRC"
OUT = Path(os.environ.get("CHAT_OUTPUT", REPO / "output" / "chatgarment_chat")).resolve()
WEB = HERE / "chat_web"
GC_PYTHON = Path(os.environ.get("GC_PYTHON", REPO / ".venv" / "Scripts" / "python.exe"))
N_FLOATS = len(json.loads((CG_ROOT / "docs" / "all_float_paths.json").read_text(encoding="utf-8")))

# 作者的問句（照片兩步驟、文字生成、修改）
Q_DESCRIBE = "Can you describe the geometry features of the garments worn by the model in the Json format?"
Q_PATTERN = "Can you estimate the sewing pattern code based on the image and Json format garment geometry description?"
Q_TEXTGEN = "Can you estimate the outfit sewing pattern code based on the Json format garment geometry description?"
Q_EDIT = ("Adjust the old sewing pattern according to the text descriptions.\nThe old garment sewing pattern is: \n"
          "{old}. \nAnd the text descriptions are: \n{desc}.")


class Model:
    """包一層 infer.py：載入一次，之後每次 ask() 問一句（可以不給照片，用作者的全黑圖代替）。"""

    def __init__(self, mode):
        sys.argv = [sys.argv[0]]  # infer.py 在匯入時不讀參數，這裡只是保險
        import torch
        import infer
        self.torch, self.infer = torch, infer
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        if mode == "auto":
            mode = "bf16" if total >= infer.BF16_FULL_GPU_GIB else "4bit"
        if mode == "bf16" and total < infer.BF16_FULL_GPU_GIB:
            raise SystemExit(f"顯示卡只有 {total:.1f} GB，bf16 放不下，請用 --mode 4bit")
        self.mode = mode
        t = time.time()
        self.tok, seg = infer.load_tokenizer()
        model = infer.build_model(seg, len(self.tok))
        sd, _ = infer.merged_state_dict(infer.CKPT)
        model.load_state_dict(sd, strict=False, assign=True)
        model = model.to(torch.bfloat16).eval()
        self.model = infer.quantize_to_gpu(model) if mode == "4bit" else model.to("cuda")
        self.processor = self.model.get_model().get_vision_tower().image_processor
        self.black = infer.preprocess(CG_ROOT / "docs" / "images" / "black_img.jpg", self.processor)
        self.name = f"{torch.cuda.get_device_name(0)}（{total:.0f} GB，{mode}）"
        print(f"模型載入完成 {time.time() - t:.0f} 秒：{self.name}", flush=True)

    def image(self, path):
        return self.infer.preprocess(path, self.processor) if path else self.black

    def ask(self, image, question, max_new_tokens=2048):
        from llava import conversation as conversation_lib
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.mm_utils import tokenizer_image_token
        torch = self.torch
        conv = conversation_lib.conv_templates["v1"].copy()
        conv.messages = []
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        ids = tokenizer_image_token(conv.get_prompt(), self.tok, return_tensors="pt").unsqueeze(0).to("cuda")
        img = image.unsqueeze(0).to("cuda", dtype=torch.bfloat16)
        with torch.no_grad():
            out, floats, _ = self.model.evaluate(img, img, ids, max_new_tokens=max_new_tokens, tokenizer=self.tok)
        text = self.tok.decode(out[0, 1:], skip_special_tokens=False).strip().replace("</s>", "")
        text = text.replace("[STARTS]", "").replace("[SEG]", "").replace("[ENDS]", "")
        return text, (None if floats is None else floats.float().cpu().numpy())


class MockModel:
    """沒有顯示卡時測網頁用：不跑 AI，照問題種類回傳 results/chatgarment_ai/bf16_3090 裡現成的輸出。"""

    name = "測試模式（--mock，回傳現成結果，不是真的推論）"
    mode = "mock"
    SAMPLE = REPO / "results" / "chatgarment_ai" / "bf16_3090" / "g03_blue_hoodie"

    def image(self, path):
        return path

    def ask(self, image, question, max_new_tokens=2048):
        import numpy as np
        time.sleep(1.0)
        txt = (self.SAMPLE / "output.txt").read_text(encoding="utf-8")
        desc, pattern = txt.split("[描述]")[1].split("[版型]")
        floats = np.load(self.SAMPLE / "float_preds.npy")
        if question == Q_DESCRIBE:
            return desc.strip(), None
        if question.startswith(Q_PATTERN) or question.startswith(Q_TEXTGEN):
            return pattern.strip(), floats
        if question.startswith("Adjust the old sewing pattern"):
            upper = json.loads((self.SAMPLE / "output.json").read_text(encoding="utf-8"))["upperbody_garment"]
            return json.dumps(upper).replace('"', "'"), floats.reshape(2, -1)[0]
        return "（測試模式沒有真的模型，自由提問不會有回答）", None


def repair(text):
    """模型輸出的 JSON 常常少括號，用作者的 repair_json 修；沒有完整 ChatGarment 原始碼時（--mock）用簡單的解析。"""
    sys.path.insert(0, str(CG_ROOT))
    try:
        from llava.json_fixer import repair_json
    except ImportError:
        import ast
        return ast.literal_eval(text.replace("null", "None").replace("true", "True").replace("false", "False"))
    return repair_json(text, return_objects=True)


class Session:
    """一次問答的資料夾。"""

    def __init__(self, kind):
        self.id = time.strftime("%m%d_%H%M%S") + f"_{kind}"
        self.dir = OUT / self.id
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, describe, pattern_text, floats, meta):
        import numpy as np
        pred = repair(pattern_text)
        if not isinstance(pred, dict):
            raise ValueError("模型輸出的版型不是 JSON")
        if not any(k in pred for k in ("upperbody_garment", "wholebody_garment")):
            # 修改衣服時，模型直接輸出一件衣服的款式（作者的腳本當作連身款、取最後 76 個數值）
            pred = {"wholebody_garment": pred}
            if floats is not None:
                floats = floats.reshape(-1)[-N_FLOATS:]
        (self.dir / "output.txt").write_text(f"[描述]\n{describe}\n\n[版型]\n{pattern_text}\n", encoding="utf-8")
        (self.dir / "output.json").write_text(json.dumps(pred, ensure_ascii=False, indent=2), encoding="utf-8")
        if floats is not None:
            np.save(self.dir / "float_preds.npy", floats)
        (self.dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        return pred


def save_upload(session, b64):
    """網頁送來的照片（data URL）存成檔案，回傳路徑。"""
    if not b64:
        return None
    raw = base64.b64decode(b64.split(",", 1)[-1])
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    path = session.dir / "input.png"
    img.save(path)
    return path


def worker(*args):
    """呼叫 chat_worker.py（GarmentCode 環境），回傳它最後一行印出的 JSON。"""
    cmd = [str(GC_PYTHON), "-W", "ignore", str(HERE / "chat_worker.py"), *map(str, args)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
    lines = [l for l in p.stdout.splitlines() if l.startswith("{")]
    if p.returncode != 0 or not lines:
        raise RuntimeError((p.stderr or p.stdout)[-2000:])
    return json.loads(lines[-1]), p.stdout


class App:
    def __init__(self, model):
        self.model = model
        self.lock = threading.Lock()  # 顯示卡一次只跑一個問題

    def ask(self, image, question, log):
        with self.lock:
            t = time.time()
            text, floats = self.model.ask(image, question)
        log.append({"question": question, "answer": text, "seconds": round(time.time() - t, 1),
                    "floats": None if floats is None else [round(float(v), 4) for v in floats.reshape(-1)]})
        return text, floats

    def run(self, req):
        mode = req["mode"]
        s = Session(mode)
        log = []
        photo = save_upload(s, req.get("image"))
        image = self.model.image(photo)
        meta = {"mode": mode, "model": self.model.name, "request": {k: v for k, v in req.items() if k != "image"}}
        result = {"id": s.id, "log": log, "has_image": photo is not None}

        if mode == "photo":        # 作者的兩步驟：先描述、再根據描述給版型
            if photo is None:
                raise ValueError("照片 → 衣服要上傳照片")
            describe, _ = self.ask(image, Q_DESCRIBE, log)
            fixed = describe.replace("upper_garment", "upperbody_garment").replace("lower_garment", "lowerbody_garment")
            pattern, floats = self.ask(image, Q_PATTERN + "\n" + fixed, log)
            result["pred"] = s.save_result(describe, pattern, floats, meta)
        elif mode == "describe":   # 只問第一步
            if photo is None:
                raise ValueError("描述衣服要上傳照片")
            describe, _ = self.ask(image, Q_DESCRIBE, log)
            result["describe"] = repair(describe)
        elif mode == "text":       # 文字 → 衣服：結構化描述（作者用 GPT-4o 從一句話轉出來的格式）
            outfit = req["garments"]   # {"upperbody garment": {"garment_name":…, "geometry_styles":{…}}, …}
            desc = json.dumps(outfit).replace('"', "'")
            pattern, floats = self.ask(self.model.image(None), Q_TEXTGEN + "\n" + desc, log)
            describe = {k.replace(" ", "_"): v for k, v in outfit.items()}  # 給修正規則讀（和照片第一步的格式相同）
            result["pred"] = s.save_result(repr(describe), pattern, floats, meta)
        elif mode == "edit":       # 修改：舊款式（簡化成文字）＋要改的部位
            base = OUT / req["base"]
            design = base / f"design_{req.get('part', 'upper')}.yaml"
            if not design.exists():
                raise ValueError(f"{req['base']} 還沒有 {design.name}，請先按「產生版片」")
            old = worker("simplify", design)[0]["text"]
            desc = json.dumps(req["edits"]).replace('"', "'")
            pattern, floats = self.ask(self.model.image(None), Q_EDIT.format(old=old, desc=desc), log)
            meta["base"] = req["base"]
            shutil.copy(base / "output.txt", s.dir / "base_output.txt")
            describe = (base / "output.txt").read_text(encoding="utf-8").split("[描述]")[1].split("[版型]")[0].strip()
            result["pred"] = s.save_result(describe, pattern, floats, meta)
        elif mode == "free":       # 自由提問：看它還會回答什麼
            text, floats = self.ask(image, req["question"], log)
            result["answer"] = text
        else:
            raise ValueError(f"不認得的 mode：{mode}")
        (s.dir / "log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
        return result

    def build(self, req):
        args = ["build", OUT / req["id"]]
        if req.get("sew"):
            args.append("--sew")
        if req.get("raw"):
            args.append("--raw")
        if req.get("lower"):
            args.append("--lower")
        out, stdout = worker(*args)
        out["stdout"] = stdout[-4000:]
        return out

    def history(self):
        items = []
        for d in sorted(OUT.glob("*"), reverse=True)[:50]:
            meta = d / "meta.json"
            if meta.exists():
                m = json.loads(meta.read_text(encoding="utf-8"))
                items.append({"id": d.name, "mode": m.get("mode"), "has_image": (d / "input.png").exists(),
                              "built": sorted(p.name for p in d.glob("design_*.yaml"))})
        return items


def make_handler(app):
    files = {"/": WEB / "index.html"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt, *args):
            pass

        def send(self, code, body, ctype="application/json; charset=utf-8"):
            if isinstance(body, (dict, list)):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = unquote(urlparse(self.path).path)
            if path in files:
                return self.send(200, files[path].read_bytes(), "text/html; charset=utf-8")
            if path == "/api/info":
                return self.send(200, {"model": app.model.name, "mode": app.model.mode})
            if path == "/api/history":
                return self.send(200, app.history())
            if path == "/body.obj":  # 縫衣服用的身體（GarmentCode mean_male，公尺）
                return self.send(200, (GC_ROOT / "assets" / "bodies" / "mean_male.obj").read_bytes(), "text/plain")
            if path.startswith("/files/"):
                target = (OUT / path[len("/files/"):]).resolve()
                if OUT in target.parents and target.is_file():
                    ctype = {".png": "image/png", ".obj": "text/plain", ".json": "application/json",
                             ".yaml": "text/plain; charset=utf-8", ".txt": "text/plain; charset=utf-8",
                             ".svg": "image/svg+xml"}.get(target.suffix, "application/octet-stream")
                    return self.send(200, target.read_bytes(), ctype)
            self.send(404, {"error": "找不到"})

        def do_POST(self):
            path = urlparse(self.path).path
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if path == "/api/run":
                    return self.send(200, app.run(req))
                if path == "/api/build":
                    return self.send(200, app.build(req))
                self.send(404, {"error": "找不到"})
            except Exception as e:  # 錯誤原樣顯示在網頁上，方便看模型輸出哪裡壞掉
                import traceback
                traceback.print_exc()
                self.send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["auto", "4bit", "bf16"], default="auto")
    ap.add_argument("--port", type=int, default=8800)
    ap.add_argument("--mock", action="store_true", help="不載入模型，用現成結果測網頁")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if not args.mock:
        sys.path.insert(0, str(HERE))
    app = App(MockModel() if args.mock else Model(args.mode))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"開啟 http://127.0.0.1:{args.port}/（{app.model.name}）", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
