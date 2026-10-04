import { initTryOnViewer, loadTryOnGlb } from "/web_test/tryon.js";

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function fillProductId(productId) {
  document.getElementById("tryonProductId").value = productId;
}

function renderResults(container, items) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    container.textContent = "沒有結果";
    return;
  }

  for (const item of items) {
    const div = document.createElement("div");
    div.className = "card";
    div.title = "點擊帶入試穿 product_id";
    const score = typeof item.score === "number" ? item.score.toFixed(4) : "-";
    const imageUrl = item.image_url || "";
    div.innerHTML = `
      <img src="${imageUrl}" alt="">
      <div>${item.name || ""}</div>
      <div class="meta">id: ${item.product_id} · score: ${score}</div>
    `;
    div.addEventListener("click", () => fillProductId(item.product_id));
    container.appendChild(div);
  }
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return res.json();
}

document.getElementById("healthBtn").addEventListener("click", async () => {
  const res = await fetch("/api/health");
  setText("healthResult", JSON.stringify(await res.json(), null, 2));
});

document.getElementById("uploadForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData();
  formData.append("name", document.getElementById("productName").value);
  formData.append("category", document.getElementById("productCategory").value);
  formData.append("garment_template_id", document.getElementById("garmentTemplateId").value);
  formData.append("image", document.getElementById("productImage").files[0]);

  const res = await fetch("/api/products/upload", { method: "POST", body: formData });
  const data = await res.json();
  setText("uploadResult", JSON.stringify(data, null, 2));
  if (data.success && data.data && data.data.product_id) {
    fillProductId(data.data.product_id);
  }
});

async function runSearch() {
  const container = document.getElementById("searchResults");
  container.textContent = "搜尋中...";
  const data = await postJson("/api/recommend", {
    query_text: document.getElementById("queryText").value,
    query_image: null,
    top_k: 5,
  });
  if (!data.success) {
    container.textContent = JSON.stringify(data);
    return;
  }
  renderResults(container, data.data.items);
}

document.getElementById("searchBtn").addEventListener("click", runSearch);

async function runImageSearch() {
  const container = document.getElementById("imageSearchResults");
  container.textContent = "搜尋中...";
  const data = await postJson("/api/recommend", {
    query_text: null,
    query_image: document.getElementById("queryImagePath").value,
    top_k: 5,
  });
  if (!data.success) {
    container.textContent = JSON.stringify(data);
    return;
  }
  renderResults(container, data.data.items);
}

document.getElementById("imageSearchBtn").addEventListener("click", runImageSearch);

async function loadProductPicker() {
  const container = document.getElementById("productPicker");
  const res = await fetch("/api/products");
  const data = await res.json();
  if (!data.success) {
    container.textContent = "載入商品失敗";
    return;
  }
  container.innerHTML = "";
  for (const item of data.data.items) {
    const label = document.createElement("label");
    label.className = "picker-item";
    label.innerHTML = `
      <input type="checkbox" value="${item.product_id}">
      <img src="${item.image_url || ""}" alt="">
      <span>${item.name}（${item.category}）</span>
    `;
    container.appendChild(label);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  loadProductPicker();
  initTryOnViewer(document.getElementById("tryonCanvas"));
});

document.getElementById("outfitCompleteBtn").addEventListener("click", async () => {
  const container = document.getElementById("outfitCompleteResults");
  const checked = Array.from(document.querySelectorAll("#productPicker input:checked"));
  const productIds = checked.map((el) => Number(el.value));
  const targetCategory = document.getElementById("targetCategory").value.trim();

  if (productIds.length === 0) {
    container.textContent = "請至少勾選 1 件商品";
    return;
  }
  if (!targetCategory) {
    container.textContent = "請輸入想找的分類";
    return;
  }

  container.textContent = "搜尋中...";
  const data = await postJson("/api/outfit/complete", {
    product_ids: productIds,
    target_category: targetCategory,
    top_k: 5,
  });
  if (!data.success) {
    container.textContent = JSON.stringify(data);
    return;
  }
  renderResults(container, data.data.items);
});

document.getElementById("tryonForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const linkEl = document.getElementById("tryonLink");
  linkEl.innerHTML = "";
  initTryOnViewer(document.getElementById("tryonCanvas"));
  setText("tryonResult", "產生人體與衣服中，請稍候…");

  const data = await postJson("/api/tryon/simulate", {
    product_id: Number(document.getElementById("tryonProductId").value),
    size: document.getElementById("tryonSize").value || "M",
    body: {
      height_cm: Number(document.getElementById("heightCm").value),
      chest_cm: Number(document.getElementById("chestCm").value),
      waist_cm: Number(document.getElementById("waistCm").value),
      hip_cm: Number(document.getElementById("hipCm").value),
    },
  });

  setText("tryonResult", JSON.stringify(data, null, 2));
  const url = data.data && data.data.result_glb_url;
  if (data.success && url) {
    const a = document.createElement("a");
    a.href = url;
    a.textContent = `下載 / 開啟 GLB：${url}`;
    linkEl.appendChild(a);
    try {
      await loadTryOnGlb(url);
    } catch (err) {
      setText("tryonResult", (data && JSON.stringify(data, null, 2)) + "\n載入畫布失敗：" + (err.message || err));
    }
  }
});
