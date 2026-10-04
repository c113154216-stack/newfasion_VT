// 負責人：林宇鑫（尺寸推薦）
//
// 尺寸判定：成衣尺寸 − 身體尺寸，依計畫書表 2-1 分成偏緊／合身／偏寬。
// 建議號碼的規則（與組員討論後）：每個部位都 ≥ 合身下限的號碼中，取最小的那一個。

export const PART_LABELS = { chest: "胸圍", shoulder: "肩寬", length: "衣長" };

// [合身下限, 合身上限]，單位 cm
const FIT_RANGES = {
  slim: { chest: [4, 10], shoulder: [0, 3], length: [-2, 2] },
  loose: { chest: [10, 18], shoulder: [0, 3], length: [-2, 2] },
};

// 長版外套的衣長本來就遠超過身體參考長度，不列入判定
function rangesFor(product) {
  const ranges = { ...FIT_RANGES[product.fit || "slim"] };
  if (product.longLength) delete ranges.length;
  return ranges;
}

// 肩寬與衣長參考值：正式版由紀泓宇的 SMPL-X 本人模型量測，這裡先用身高估算
export function deriveBody(input) {
  return {
    ...input,
    shoulder: round1(input.height * 0.255),
    lengthRef: round1(input.height * 0.39),
  };
}

function bodyValue(body, part) {
  return part === "length" ? body.lengthRef : body[part];
}

export function classify(diff, [lo, hi]) {
  if (diff < lo) return "tight";
  if (diff > hi) return "loose";
  return "fit";
}

export function evaluateSize(body, product, size) {
  const chart = product.sizeChart[size];
  const ranges = rangesFor(product);
  return Object.entries(ranges).map(([part, range]) => {
    const garment = chart[part];
    const diff = round1(garment - bodyValue(body, part));
    return { part, label: PART_LABELS[part], garment, diff, range, status: classify(diff, range) };
  });
}

export function recommend(body, product) {
  if (!product.sizeChart) return null;
  const sizes = Object.keys(product.sizeChart);
  const results = Object.fromEntries(sizes.map((s) => [s, evaluateSize(body, product, s)]));
  const ok = sizes.find((s) => results[s].every((p) => p.diff >= p.range[0]));
  return {
    sizes,
    results,
    recommended: ok ?? sizes[sizes.length - 1],
    allTight: !ok,
  };
}

export function adviceText(product, size, parts, rec) {
  if (rec.allTight) {
    return { text: `所有號碼都有部位偏緊，最大號 ${size} 仍不夠，建議改看其他款式。`, warn: true };
  }
  const tight = parts.filter((p) => p.status === "tight");
  const loose = parts.filter((p) => p.status === "loose");
  const idx = rec.sizes.indexOf(size);
  if (tight.length) {
    const next = rec.sizes[idx + 1];
    const names = tight.map((p) => p.label).join("、");
    return { text: `${names}偏緊${next ? `，在意的話改選 ${next}` : ""}。`, warn: true };
  }
  if (size === rec.recommended) {
    return {
      text: loose.length
        ? `建議 ${size}。${loose.map((p) => p.label).join("、")}會比較寬鬆。`
        : `建議 ${size}，各部位都在合身範圍內。`,
      warn: false,
    };
  }
  return { text: `${size} 不會緊，但比建議的 ${rec.recommended} 寬鬆。`, warn: false };
}

function round1(v) {
  return Math.round(v * 10) / 10;
}
