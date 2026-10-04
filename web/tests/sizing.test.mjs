// 尺寸推薦的自動測試（對應計畫書表 2-1 與「尺寸判定一致率」）。
// 跑法：node --test web/tests/
import test from "node:test";
import assert from "node:assert/strict";

import { deriveBody, recommend, evaluateSize, classify, adviceText } from "../js/sizing.js";
import { PRODUCTS } from "../js/mock-data.js";

const product = (id) => PRODUCTS.find((p) => p.id === id);
const TEE = product(1); // 修身，S/M/L/XL
const SWEATER = product(9); // 寬鬆，S/M/L
const COAT = product(5); // 寬鬆長版大衣

const statusOf = (parts, part) => parts.find((p) => p.part === part).status;

test("表 2-1 區間邊界：等於下限、上限都算合身", () => {
  assert.equal(classify(4, [4, 10]), "fit");
  assert.equal(classify(10, [4, 10]), "fit");
  assert.equal(classify(3.9, [4, 10]), "tight");
  assert.equal(classify(10.1, [4, 10]), "loose");
});

test("修身 T 恤：每個部位都夠大的最小號碼是 M", () => {
  // 身高 170 → 肩寬 43.4、衣長參考 66.3
  const body = deriveBody({ height: 170, weight: 62, chest: 92, waist: 76, hip: 94 });
  const rec = recommend(body, TEE);
  assert.equal(rec.recommended, "M");
  assert.equal(rec.allTight, false);

  const parts = evaluateSize(body, TEE, "M");
  assert.equal(statusOf(parts, "chest"), "fit"); // 96 − 92 = +4，剛好在下限
  assert.equal(statusOf(parts, "shoulder"), "fit"); // 44 − 43.4 = +0.6
  assert.equal(statusOf(parts, "length"), "fit"); // 68 − 66.3 = +1.7
});

test("肩寬不夠時往上跳一號：胸圍 M 夠，但肩寬只有 L 夠", () => {
  // 預設身體：身高 175 → 肩寬 44.6，M 的 44 不夠
  const body = deriveBody({ height: 175, weight: 68, chest: 92, waist: 78, hip: 96 });
  const rec = recommend(body, TEE);
  assert.equal(rec.recommended, "L");

  const m = evaluateSize(body, TEE, "M");
  assert.equal(statusOf(m, "shoulder"), "tight");
  assert.equal(statusOf(m, "chest"), "fit");
});

test("小號身材選 S；衣長偏長只算偏寬，不會讓號碼往下跳", () => {
  // 身高 160 → 肩寬 40.8、衣長參考 62.4
  const body = deriveBody({ height: 160, weight: 50, chest: 82, waist: 66, hip: 88 });
  const rec = recommend(body, TEE);
  assert.equal(rec.recommended, "S");
  const parts = evaluateSize(body, TEE, "S");
  assert.equal(statusOf(parts, "length"), "loose"); // 66 − 62.4 = +3.6
});

test("所有號碼都不夠大：給最大號並標記全部偏緊", () => {
  const body = deriveBody({ height: 180, weight: 110, chest: 115, waist: 105, hip: 112 });
  const rec = recommend(body, TEE);
  assert.equal(rec.recommended, "XL");
  assert.equal(rec.allTight, true);
  const advice = adviceText(TEE, "XL", evaluateSize(body, TEE, "XL"), rec);
  assert.equal(advice.warn, true);
});

test("寬鬆版胸圍門檻是 +10，修身可以的 +8 在寬鬆版算偏緊", () => {
  const body = deriveBody({ height: 170, weight: 62, chest: 92, waist: 76, hip: 94 });
  const rec = recommend(body, SWEATER);
  assert.equal(rec.recommended, "M"); // S 只多 8 cm，不夠；M 多 16 cm
  assert.equal(statusOf(evaluateSize(body, SWEATER, "S"), "chest"), "tight");
});

test("長版大衣不比衣長", () => {
  const body = deriveBody({ height: 170, weight: 62, chest: 92, waist: 76, hip: 94 });
  const parts = evaluateSize(body, COAT, "M");
  assert.equal(parts.some((p) => p.part === "length"), false);
});

test("沒有尺寸表（上傳的平面圖）不給建議", () => {
  const body = deriveBody({ height: 170, weight: 62, chest: 92, waist: 76, hip: 94 });
  assert.equal(recommend(body, { ...TEE, sizeChart: null }), null);
});

test("建議文字：選到偏緊的號碼會提示改大一號", () => {
  const body = deriveBody({ height: 175, weight: 68, chest: 92, waist: 78, hip: 96 });
  const rec = recommend(body, TEE);
  const advice = adviceText(TEE, "M", evaluateSize(body, TEE, "M"), rec);
  assert.equal(advice.warn, true);
  assert.match(advice.text, /肩寬偏緊/);
  assert.match(advice.text, /改選 L/);
});

test("所有示範商品：建議號碼一定符合「每個部位 ≥ 合身下限」，且是最小的那一個", () => {
  const bodies = [150, 160, 170, 180, 190].flatMap((height) =>
    [80, 90, 100, 110].map((chest) => deriveBody({ height, weight: 60, chest, waist: chest - 14, hip: chest + 4 })),
  );
  for (const p of PRODUCTS.filter((x) => x.sizeChart)) {
    for (const body of bodies) {
      const rec = recommend(body, p);
      const enough = (size) => evaluateSize(body, p, size).every((x) => x.diff >= x.range[0]);
      if (rec.allTight) {
        assert.equal(rec.sizes.some(enough), false, `${p.name} 應該沒有夠大的號碼`);
        continue;
      }
      assert.ok(enough(rec.recommended), `${p.name} 建議 ${rec.recommended} 卻有部位不夠大`);
      const smaller = rec.sizes.slice(0, rec.sizes.indexOf(rec.recommended));
      assert.equal(smaller.some(enough), false, `${p.name} 有更小的號碼也夠大`);
    }
  }
});
