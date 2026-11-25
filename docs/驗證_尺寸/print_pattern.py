"""③ 版片 1:1 列印檔（A4 分頁拼貼）：z06 M 的半片後片 + 前袖片，印出來用尺量。
第 1 頁：10 cm 校正方框（先量這個，確認印表機是「實際大小 / 100%」沒縮放）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, r"C:\Users\User\Desktop\newfashion_VT\flows\size_chart")
import spec_to_pattern as sp  # noqa: E402

plt.rcParams["font.family"] = ["Microsoft JhengHei", "sans-serif"]
gid, size = (sys.argv[1:3] if len(sys.argv) > 2 else ("z06", "M"))
R = Path(r"C:\Users\User\Desktop\newfashion_VT\output")
name = f"specb_{gid}_{size}"
P = json.loads((R / "chatgarment" / name / f"{name}_specification.json").read_text(encoding="utf-8"))["pattern"]["panels"]
OUT = Path(__file__).with_name(f"版片1比1_{gid}_{size}.pdf")

A4W, A4H, MG = 21.0, 29.7, 1.0  # cm
TW, TH = A4W - 2 * MG, A4H - 2 * MG  # 每頁印得到的範圍 19 × 27.7


def outline(pn):
    _, edges = sp.sample_panel(P[pn], n=60)
    return edges


def page(pdf, x0, y0, draw, title):
    fig = plt.figure(figsize=(A4W / 2.54, A4H / 2.54))
    ax = fig.add_axes([MG / A4W, MG / A4H, TW / A4W, TH / A4H])
    ax.set_xlim(x0, x0 + TW)
    ax.set_ylim(y0, y0 + TH)
    ax.axis("off")
    # 5 cm 格線（座標跨頁連續，拼貼時對齊用）
    for gx in np.arange(np.floor(x0 / 5) * 5, x0 + TW + 5, 5):
        ax.axvline(gx, color="#cccccc", lw=0.4)
    for gy in np.arange(np.floor(y0 / 5) * 5, y0 + TH + 5, 5):
        ax.axhline(gy, color="#cccccc", lw=0.4)
    ax.add_patch(plt.Rectangle((x0, y0), TW, TH, fill=False, ec="#999999", lw=0.6, ls="--"))
    draw(ax)
    # 頁邊：5 cm 比例尺 + 頁碼
    pa = fig.add_axes([0, 0, 1, 1])
    pa.set_xlim(0, A4W)
    pa.set_ylim(0, A4H)
    pa.axis("off")
    pa.plot([MG, MG + 5], [0.45, 0.45], color="k", lw=1.5)
    pa.text(MG + 5.3, 0.45, "← 這條是 5 cm", va="center", fontsize=7)
    pa.text(A4W - MG, 0.45, title, ha="right", va="center", fontsize=7)
    pdf.savefig(fig)
    plt.close(fig)


def tiles(pdf, xmin, ymin, xmax, ymax, draw, label):
    nx = int(np.ceil((xmax - xmin) / TW))
    ny = int(np.ceil((ymax - ymin) / TH))
    for r in range(ny - 1, -1, -1):
        for c in range(nx):
            page(pdf, xmin + c * TW, ymin + r * TH, draw, f"{label}　第 {ny - r} 列、第 {c + 1} 欄（共 {ny}×{nx} 頁）")


with PdfPages(OUT) as pdf:
    # 校正頁
    fig = plt.figure(figsize=(A4W / 2.54, A4H / 2.54))
    pa = fig.add_axes([0, 0, 1, 1])
    pa.set_xlim(0, A4W)
    pa.set_ylim(0, A4H)
    pa.axis("off")
    pa.add_patch(plt.Rectangle((5.5, 12), 10, 10, fill=False, ec="k", lw=1.2))
    pa.text(10.5, 17, "10 cm × 10 cm", ha="center", va="center", fontsize=14)
    pa.text(A4W / 2, 26.5, f"{gid} {size} 版片 1:1 列印", ha="center", fontsize=18)
    pa.text(2, 9.5, "1. 列印時選「實際大小 / 100%」，不要選「符合頁面」。\n"
                    "2. 先量上面的方框：兩邊都要是 10 cm（±1 mm），不對就是印表機有縮放。\n"
                    "3. 後面每頁有淺灰 5 cm 格線，座標跨頁連續：剪掉虛線外的白邊，照格線拼起來。\n"
                    "4. 拿捲尺量版片上標的紅線，和頁上寫的數字比。", fontsize=10, va="top", linespacing=1.8)
    pdf.savefig(fig)
    plt.close(fig)

    # 半片後片（right_btorso，x 從 −29.4 到 0，0 是後中心）
    bk = outline("right_btorso")
    bv = np.asarray(P["right_btorso"]["vertices"], float)
    top = bv[bv[:, 1] > bv[:, 1].max() - 8]
    cn = top[np.abs(top[:, 0]).argmax()]
    L = np.ptp(bv[:, 1])

    def draw_back(ax):
        for e in bk:
            ax.plot(e[:, 0], e[:, 1], color="k", lw=1.2)
        ax.plot([cn[0], 0], [cn[1], cn[1]], color="crimson", lw=1.2)
        ax.plot(cn[0], cn[1], "o", color="crimson", ms=4)
        ax.text(cn[0] / 2, cn[1] - 1.2, f"肩點 → 後中心：{abs(cn[0]):.1f} cm（×2 = 肩寬 {2 * abs(cn[0]):.1f}）",
                color="crimson", ha="center", fontsize=8)
        xl = -3
        ax.plot([xl, xl], [bv[:, 1].min(), bv[:, 1].max()], color="darkgreen", lw=1.2)
        for yy in (bv[:, 1].min(), bv[:, 1].max()):
            ax.plot([xl - 0.6, xl + 0.6], [yy, yy], color="darkgreen", lw=1.2)
        for yy in np.arange(5, L, 20):
            ax.text(xl - 0.5, yy, f"衣長 {L:.1f} cm（肩頸點→下襬）", color="darkgreen", rotation=90,
                    ha="right", va="bottom", fontsize=8)
        ax.text(-15, 25, "半片後片\n（右後片，右邊是後中心）", ha="center", fontsize=10, color="#555555")

    xs = np.concatenate([e[:, 0] for e in bk])
    ys = np.concatenate([e[:, 1] for e in bk])
    tiles(pdf, xs.min() - 1, ys.min() - 1, xs.max() + 1, ys.max() + 1, draw_back, "後片")

    # 前袖片（right_sleeve_f）
    sl = outline("right_sleeve_f")
    sv = np.asarray(P["right_sleeve_f"]["vertices"], float)

    def draw_sleeve(ax):
        for e in sl:
            ax.plot(e[:, 0], e[:, 1], color="k", lw=1.2)
        y = sv[:, 1].max()
        ax.plot([sv[:, 0].min(), sv[:, 0].max()], [y + 1, y + 1], color="navy", lw=1.2)
        ax.text(np.mean(sv[:, 0]), y + 1.5, f"袖長 {np.ptp(sv[:, 0]):.1f} cm（袖口 → 肩點，沿袖子上緣）",
                color="navy", ha="center", fontsize=9)
        ax.text(np.mean(sv[:, 0]), sv[:, 1].min() + 8, "前袖片", ha="center", fontsize=10, color="#555555")

    xs = np.concatenate([e[:, 0] for e in sl])
    ys = np.concatenate([e[:, 1] for e in sl])
    tiles(pdf, xs.min() - 1, ys.min() - 1, xs.max() + 1, ys.max() + 3, draw_sleeve, "袖片")

print(OUT)
