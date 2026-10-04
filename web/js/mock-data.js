// 假資料：之後換成趙丞章的 API（GET /api/products 等）。
// 尺寸單位都是公分。上衣尺寸表依計畫書至少要有胸圍、肩寬、衣長。

const IMG = "assets/products/";

export const CATEGORIES = [
  { id: "all", label: "全部" },
  { id: "top", label: "上衣" },
  { id: "outer", label: "外套" },
];

// fit：slim（修身）/ loose（寬鬆），決定合身判定用哪組區間
// sleeve：short / long，給畫面上的衣服用
// tryOn=false：只出現在搭配建議，目前範圍（上半身）不能試穿
export const PRODUCTS = [
  {
    id: 1, name: "白色短袖 T 恤", category: "top", fit: "slim", sleeve: "short",
    image: IMG + "01_white_tee.jpg", color: "#f2f2f0", price: 390, tags: ["白", "短袖", "圓領"],
    sizeChart: {
      S: { chest: 88, shoulder: 42, length: 66 },
      M: { chest: 96, shoulder: 44, length: 68 },
      L: { chest: 104, shoulder: 46, length: 70 },
      XL: { chest: 112, shoulder: 48, length: 72 },
    },
  },
  {
    id: 2, name: "海軍藍牛津襯衫", category: "top", fit: "slim", sleeve: "long",
    image: IMG + "02_navy_shirt.jpg", color: "#1f3155", price: 890, tags: ["藍", "長袖", "襯衫"],
    sizeChart: {
      S: { chest: 92, shoulder: 43, length: 70 },
      M: { chest: 100, shoulder: 45, length: 72 },
      L: { chest: 108, shoulder: 47, length: 74 },
    },
  },
  {
    id: 8, name: "橄欖綠 Polo 衫", category: "top", fit: "slim", sleeve: "short",
    image: IMG + "08_olive_polo.jpg", color: "#556b2f", price: 690, tags: ["綠", "短袖", "polo"],
    sizeChart: {
      S: { chest: 90, shoulder: 42, length: 67 },
      M: { chest: 98, shoulder: 44, length: 69 },
      L: { chest: 106, shoulder: 46, length: 71 },
    },
  },
  {
    id: 9, name: "奶油色針織衫", category: "top", fit: "loose", sleeve: "long",
    image: IMG + "09_cream_sweater.jpg", color: "#efe6d4", price: 990, tags: ["米", "長袖", "針織"],
    sizeChart: {
      S: { chest: 100, shoulder: 45, length: 64 },
      M: { chest: 108, shoulder: 47, length: 66 },
      L: { chest: 116, shoulder: 49, length: 68 },
    },
  },
  {
    id: 6, name: "灰色連帽上衣", category: "outer", fit: "loose", sleeve: "long",
    image: IMG + "06_grey_hoodie.jpg", color: "#80808a", price: 1190, tags: ["灰", "長袖", "帽 T"],
    sizeChart: {
      S: { chest: 104, shoulder: 46, length: 66 },
      M: { chest: 112, shoulder: 48, length: 68 },
      L: { chest: 120, shoulder: 50, length: 70 },
    },
  },
  {
    id: 5, name: "米色長版大衣", category: "outer", fit: "loose", sleeve: "long", longLength: true,
    image: IMG + "05_beige_coat.jpg", color: "#cdb995", price: 2490, tags: ["米", "長袖", "大衣"],
    sizeChart: {
      S: { chest: 102, shoulder: 45, length: 92 },
      M: { chest: 110, shoulder: 47, length: 96 },
      L: { chest: 118, shoulder: 49, length: 100 },
    },
  },
  { id: 3, name: "卡其直筒長褲", category: "bottom", tryOn: false, image: IMG + "03_khaki_pants.jpg", tags: ["卡其", "長褲"] },
  { id: 4, name: "黑色運動短褲", category: "bottom", tryOn: false, image: IMG + "04_black_shorts.jpg", tags: ["黑", "短褲"] },
  { id: 10, name: "炭灰縮口褲", category: "bottom", tryOn: false, image: IMG + "10_charcoal_joggers.jpg", tags: ["灰", "長褲"] },
];

// 假的搭配規則：之後換成 POST /api/outfit/complete
export const OUTFIT_RULES = {
  top: [{ ids: [3, 10, 4], tag: "下身" }, { ids: [6, 5], tag: "外層" }],
  outer: [{ ids: [1, 9], tag: "內搭" }, { ids: [3, 10], tag: "下身" }],
};

export const DEFAULT_BODY = { height: 175, weight: 68, chest: 92, waist: 78, hip: 96 };
