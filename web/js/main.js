import { CATEGORIES, PRODUCTS, OUTFIT_RULES, DEFAULT_BODY } from "./mock-data.js";
import { deriveBody, recommend, evaluateSize, adviceText } from "./sizing.js";
import { Viewer } from "./viewer.js";
import { PoseSource } from "./pose.js";

const $ = (id) => document.getElementById(id);

const state = {
  products: [...PRODUCTS],
  category: "all",
  query: "",
  body: deriveBody(DEFAULT_BODY),
  product: null,
  size: null,
  rec: null,
  latencyMs: null,
};

// ---------- 3D 人偶與鏡頭 ----------
const viewer = new Viewer($("viewer"));
viewer.setBody(state.body);

const camPane = $("camPane");
const pose = new PoseSource({
  video: $("video"),
  canvas: $("poseCanvas"),
  onPose: (angles, meta) => {
    viewer.setPose(angles);
    state.latencyMs = meta.latencyMs;
  },
  onStatus: ({ text, kind, live }) => {
    const el = $("trackStatus");
    el.textContent = text;
    el.className = `chip chip-bl ${kind || "muted"}`;
    camPane.classList.toggle("live", !!live);
  },
});
pose.start();

$("startCam").addEventListener("click", async () => {
  const btn = $("startCam");
  btn.disabled = true;
  const ok = await pose.startCamera();
  btn.hidden = ok;
  btn.disabled = false;
});

$("toggleSkeleton").addEventListener("click", (e) => {
  const on = e.currentTarget.getAttribute("aria-pressed") !== "true";
  e.currentTarget.setAttribute("aria-pressed", String(on));
  e.currentTarget.textContent = on ? "骨架 開" : "骨架 關";
  pose.setSkeletonVisible(on);
});

$("simMode").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-mode]");
  if (!btn) return;
  for (const b of $("simMode").querySelectorAll("button")) b.classList.toggle("active", b === btn);
  viewer.setSimMode(btn.dataset.mode);
  updateMetrics();
});

// ---------- 鏡頭／人偶分隔線（可拖拉，記住比例） ----------
const CAM_RATIO_KEY = "vto.camRatio";
function setCamRatio(r) {
  const ratio = Math.max(0.2, Math.min(0.7, r));
  document.documentElement.style.setProperty("--cam-fr", `${ratio * 5}fr`);
  document.documentElement.style.setProperty("--stage-fr", `${(1 - ratio) * 5}fr`);
  return ratio;
}
try {
  const saved = parseFloat(localStorage.getItem(CAM_RATIO_KEY));
  if (saved) setCamRatio(saved);
} catch {}

const splitter = $("splitter");
splitter.addEventListener("pointerdown", (e) => {
  splitter.setPointerCapture(e.pointerId);
  splitter.classList.add("dragging");
});
splitter.addEventListener("pointermove", (e) => {
  if (!splitter.hasPointerCapture(e.pointerId)) return;
  const cam = camPane.getBoundingClientRect();
  const stage = $("stagePane").getBoundingClientRect();
  const ratio = setCamRatio((e.clientX - cam.left) / (stage.right - cam.left));
  try { localStorage.setItem(CAM_RATIO_KEY, String(ratio)); } catch {}
});
splitter.addEventListener("pointerup", () => splitter.classList.remove("dragging"));
splitter.addEventListener("keydown", (e) => {
  if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
  const cam = camPane.getBoundingClientRect();
  const stage = $("stagePane").getBoundingClientRect();
  const now = cam.width / (stage.right - cam.left);
  setCamRatio(now + (e.key === "ArrowRight" ? 0.03 : -0.03));
});

// ---------- 窄螢幕抽屜 ----------
$("toggleRail").addEventListener("click", () => document.body.classList.toggle("rail-open"));
$("toggleSide").addEventListener("click", () => document.body.classList.toggle("side-open"));
$("scrim").addEventListener("click", () => document.body.classList.remove("rail-open", "side-open"));

// ---------- 商品列表 ----------
function renderTabs() {
  $("categoryTabs").innerHTML = CATEGORIES.map(
    (c) => `<button role="tab" data-cat="${c.id}" aria-selected="${c.id === state.category}">${c.label}</button>`,
  ).join("");
}
$("categoryTabs").addEventListener("click", (e) => {
  const btn = e.target.closest("button[data-cat]");
  if (!btn) return;
  state.category = btn.dataset.cat;
  renderTabs();
  renderProducts();
});
$("search").addEventListener("input", (e) => {
  state.query = e.target.value.trim().toLowerCase();
  renderProducts();
});

function visibleProducts() {
  return state.products.filter((p) => {
    if (p.tryOn === false) return false;
    if (state.category !== "all" && p.category !== state.category) return false;
    if (!state.query) return true;
    const hay = [p.name, ...(p.tags || [])].join(" ").toLowerCase();
    return state.query.split(/\s+/).every((q) => hay.includes(q));
  });
}

function renderProducts() {
  const list = visibleProducts();
  $("productGrid").innerHTML = list.length
    ? list
        .map(
          (p) => `
      <button class="product ${state.product && state.product.id === p.id ? "selected" : ""}" data-id="${p.id}">
        <img src="${p.image}" alt="" loading="lazy">
        <span class="name">${escapeHtml(p.name)}</span>
        <span class="meta">${p.sizeChart ? `${p.fit === "loose" ? "寬鬆" : "修身"} · NT$${p.price}` : "無尺寸表"}</span>
      </button>`,
        )
        .join("")
    : `<p class="empty">找不到符合的衣服，換個關鍵字試試。</p>`;
}
$("productGrid").addEventListener("click", (e) => {
  const btn = e.target.closest(".product");
  if (!btn) return;
  selectProduct(state.products.find((p) => String(p.id) === btn.dataset.id));
  document.body.classList.remove("rail-open");
});

function selectProduct(product) {
  state.product = product;
  state.rec = recommend(state.body, product);
  state.size = state.rec ? state.rec.recommended : null;
  viewer.setGarment(product, state.size);
  renderProducts();
  renderSize();
  renderOutfits();
  updateGarmentLabel();
  updateMetrics();
}

// ---------- 上傳平面圖（假）：沒有尺寸表，只做上身預覽 ----------
$("upload").addEventListener("change", async (e) => {
  const file = e.target.files && e.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  const color = await averageColor(url);
  const product = {
    id: `upload-${Date.now()}`,
    name: file.name.replace(/\.[^.]+$/, ""),
    category: "top",
    fit: "slim",
    sleeve: "short",
    image: url,
    color,
    sizeChart: null,
    tags: ["上傳"],
  };
  state.products.unshift(product);
  state.category = "all";
  renderTabs();
  selectProduct(product);
  e.target.value = "";
});

// ---------- 身體數值 ----------
const BODY_FIELDS = [
  { key: "height", label: "身高 cm", min: 140, max: 220 },
  { key: "weight", label: "體重 kg", min: 35, max: 160 },
  { key: "chest", label: "胸圍 cm", min: 60, max: 150 },
  { key: "waist", label: "腰圍 cm", min: 50, max: 150 },
  { key: "hip", label: "臀圍 cm", min: 60, max: 160 },
  { key: "shoulder", label: "肩寬 cm", derived: true },
  { key: "lengthRef", label: "衣長參考 cm", derived: true },
];

function renderBodyForm() {
  $("bodyForm").innerHTML = BODY_FIELDS.map(
    (f) => `
    <label class="field">
      <span>${f.label}</span>
      <input type="number" inputmode="decimal" data-key="${f.key}" value="${state.body[f.key]}"
        ${f.derived ? "readonly tabindex=\"-1\"" : `min="${f.min}" max="${f.max}" step="0.5" required`}>
    </label>`,
  ).join("");
  $("derivedHint").textContent = "虛線欄位由本人模型量測，目前先用身高估算。";
}

$("bodyForm").addEventListener("input", (e) => {
  const input = e.target;
  if (!input.dataset.key || input.readOnly || !input.checkValidity()) return;
  const raw = Object.fromEntries(
    BODY_FIELDS.filter((f) => !f.derived).map((f) => [f.key, Number(state.body[f.key])]),
  );
  raw[input.dataset.key] = Number(input.value);
  state.body = deriveBody(raw);
  for (const f of BODY_FIELDS.filter((x) => x.derived)) {
    $("bodyForm").querySelector(`[data-key="${f.key}"]`).value = state.body[f.key];
  }
  viewer.setBody(state.body);
  if (state.product) {
    state.rec = recommend(state.body, state.product);
    // 身體變了，建議號碼也跟著重算
    state.size = state.rec ? state.rec.recommended : null;
    viewer.setGarment(state.product, state.size);
  }
  renderSize();
  updateGarmentLabel();
  updateMetrics();
});

// ---------- 尺寸建議 ----------
function renderSize() {
  const box = $("sizeContent");
  const p = state.product;
  if (!p) {
    box.innerHTML = `<p class="hint">從左邊選一件衣服，這裡會顯示建議號碼和各部位鬆緊。</p>`;
    return;
  }
  if (!state.rec) {
    box.innerHTML = `<p class="advice warn">這件衣服沒有尺寸表，只提供上身預覽，不提供號碼建議。</p>`;
    return;
  }
  const rec = state.rec;
  const parts = evaluateSize(state.body, p, state.size);
  const advice = adviceText(p, state.size, parts, rec);
  box.innerHTML = `
    <div class="sizes" role="radiogroup" aria-label="號碼">
      ${rec.sizes
        .map(
          (s) => `<button class="size-btn ${s === state.size ? "active" : ""}" data-size="${s}" role="radio" aria-checked="${s === state.size}">
            ${s}${s === rec.recommended && !rec.allTight ? `<span class="star">建議</span>` : ""}</button>`,
        )
        .join("")}
    </div>
    <div class="parts">
      ${parts.map(partRow).join("")}
    </div>
    <p class="advice ${advice.warn ? "warn" : ""}">${advice.text}</p>
    <p class="hint">差值＝成衣 − 身體；綠色區間為${p.fit === "loose" ? "寬鬆版" : "修身版"}合身範圍（計畫書表 2-1）。</p>`;
}

function partRow(p) {
  const statusText = { fit: "合身", tight: "偏緊", loose: "偏寬" }[p.status];
  // 長條：把 [下限−6, 上限+6] 映射到 0–100%，綠色是合身區間
  const lo = p.range[0] - 6;
  const hi = p.range[1] + 6;
  const pos = (v) => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
  const sign = p.diff > 0 ? "+" : "";
  return `
    <span class="pname">${p.label}</span>
    <span class="pval">${p.garment}</span>
    <div class="bar" title="合身 ${p.range[0]} ~ ${p.range[1]} cm">
      <div class="range" style="left:${pos(p.range[0])}%;width:${pos(p.range[1]) - pos(p.range[0])}%"></div>
      <div class="dot ${p.status}" style="left:${pos(p.diff)}%"></div>
    </div>
    <span class="status ${p.status}">${sign}${p.diff} ${statusText}</span>`;
}

$("sizeContent").addEventListener("click", (e) => {
  const btn = e.target.closest(".size-btn");
  if (!btn) return;
  state.size = btn.dataset.size;
  // 流程：選定號碼 → 生成該號碼的衣服網格（目前是簡化外形）
  viewer.setGarment(state.product, state.size);
  renderSize();
  updateGarmentLabel();
  updateMetrics();
});

function updateGarmentLabel() {
  const p = state.product;
  $("garmentLabel").textContent = p ? `${p.name}${state.size ? ` · ${state.size}` : " · 預覽"}` : "尚未選擇衣服";
  $("garmentNote").textContent = p && !p.sizeChart ? "依你的體型預覽" : "依選定號碼生成";
}

// ---------- 搭配建議 ----------
function renderOutfits() {
  const p = state.product;
  const rules = p ? OUTFIT_RULES[p.category] : null;
  if (!rules) {
    $("outfits").innerHTML = `<span class="hint">選一件衣服後顯示搭配</span>`;
    return;
  }
  const items = rules.flatMap((r) =>
    r.ids.map((id) => ({ item: PRODUCTS.find((x) => x.id === id), tag: r.tag })).filter((x) => x.item && x.item.id !== p.id),
  );
  $("outfits").innerHTML = items
    .map(
      ({ item, tag }) => `
    <button class="outfit" data-id="${item.id}" title="${item.tryOn === false ? "目前只支援上半身試穿" : "點選試穿"}">
      <img src="${item.image}" alt="">${escapeHtml(item.name)}<span class="tag">${tag}</span>
    </button>`,
    )
    .join("");
}
$("outfits").addEventListener("click", (e) => {
  const btn = e.target.closest(".outfit");
  if (!btn) return;
  const item = PRODUCTS.find((x) => String(x.id) === btn.dataset.id);
  if (item && item.tryOn !== false) selectProduct(item);
});

// ---------- 指標 ----------
function updateMetrics() {
  const col = viewer.collisionEstimate;
  $("metrics").textContent = col === null ? "選衣服後顯示穿入估計" : `初始穿入（估）${col}%`;
}

setInterval(() => {
  $("fpsStat").textContent = `${Math.round(viewer.fps)} FPS`;
  $("latencyStat").textContent = state.latencyMs ? `延遲（估）${Math.round(state.latencyMs)} ms` : "延遲 —";
}, 500);

// ---------- 工具 ----------
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function averageColor(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const c = document.createElement("canvas");
      c.width = c.height = 24;
      const ctx = c.getContext("2d");
      ctx.drawImage(img, 0, 0, 24, 24);
      const d = ctx.getImageData(0, 0, 24, 24).data;
      let r = 0, g = 0, b = 0, n = 0;
      for (let i = 0; i < d.length; i += 4) {
        // 略過接近白色的背景
        if (d[i] > 235 && d[i + 1] > 235 && d[i + 2] > 235) continue;
        r += d[i]; g += d[i + 1]; b += d[i + 2]; n++;
      }
      resolve(n ? `rgb(${Math.round(r / n)},${Math.round(g / n)},${Math.round(b / n)})` : "#dddddd");
    };
    img.onerror = () => resolve("#999999");
    img.src = url;
  });
}

renderTabs();
renderProducts();
renderBodyForm();
renderSize();
renderOutfits();
selectProduct(PRODUCTS[0]);
