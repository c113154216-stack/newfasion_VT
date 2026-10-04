// Webcam 姿態來源：開啟鏡頭時用 MediaPipe Pose（BlazePose，33 個關鍵點），
// 還沒開鏡頭或失敗時用「示範動作」產生假的關鍵點，畫面和人偶都照樣能動。
//
// 輸出給人偶的是「螢幕上看到的」角度（畫面是鏡像，像照鏡子）：
//   left / right：螢幕左邊／右邊那隻手
//   upper：上臂相對「垂直向下」的角度，往螢幕右邊為正（弧度）
//   fore：前臂相對「垂直向下」的角度（世界角度，不是相對上臂）
//   lean：軀幹相對垂直的傾斜，往螢幕右邊為正
//   head：頭相對軀幹的傾斜

const MP_VERSION = "0.10.14";
const MP_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MP_MODEL =
  "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";

// BlazePose 骨架連線（只畫上半身＋腿的主幹，手指腳趾略過）
const BONES = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
];
const KEY_POINTS = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];
const DEMO_FRAME = { width: 640, height: 480 };

export class PoseSource {
  constructor({ video, canvas, onPose, onStatus }) {
    this.video = video;
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onPose = onPose;
    this.onStatus = onStatus;
    this.mode = "demo";
    this.showSkeleton = true;
    this.landmarker = null;
    this.lastVideoTime = -1;
    this.running = false;
    this._loop = this._loop.bind(this);
  }

  start() {
    if (this.running) return;
    this.running = true;
    this.onStatus({ text: "示範動作", kind: "muted" });
    requestAnimationFrame(this._loop);
  }

  setSkeletonVisible(visible) {
    this.showSkeleton = visible;
  }

  async startCamera() {
    this.onStatus({ text: "載入姿態模型…", kind: "muted" });
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
        audio: false,
      });
      this.video.srcObject = stream;
      await this.video.play();

      if (!this.landmarker) {
        const { FilesetResolver, PoseLandmarker } = await import(`${MP_BASE}/vision_bundle.mjs`);
        const fileset = await FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
        const create = (delegate) =>
          PoseLandmarker.createFromOptions(fileset, {
            baseOptions: { modelAssetPath: MP_MODEL, delegate },
            runningMode: "VIDEO",
            numPoses: 1,
          });
        // 有些電腦的 WebGL 不支援 GPU 推論，失敗就改用 CPU
        this.landmarker = await create("GPU").catch(() => create("CPU"));
      }
      this.mode = "camera";
      this.onStatus({ text: "追蹤中", kind: "ok", live: true });
      return true;
    } catch (err) {
      console.warn("[pose] 無法開啟鏡頭或載入模型，維持示範動作", err);
      this.mode = "demo";
      const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
      this.onStatus({ text: denied ? "鏡頭權限被拒，改用示範動作" : "鏡頭無法使用，改用示範動作", kind: "warn" });
      return false;
    }
  }

  _loop(now) {
    if (!this.running) return;
    let landmarks = null;
    let frame = DEMO_FRAME;
    let latencyMs = null;

    if (this.mode === "camera" && this.video.readyState >= 2) {
      frame = { width: this.video.videoWidth, height: this.video.videoHeight };
      if (this.video.currentTime !== this.lastVideoTime) {
        this.lastVideoTime = this.video.currentTime;
        const t0 = performance.now();
        const result = this.landmarker.detectForVideo(this.video, t0);
        const detectMs = performance.now() - t0;
        // 估計延遲：一個鏡頭影格的時間 + 偵測時間 + 一個畫面更新
        latencyMs = 1000 / 30 + detectMs + 1000 / 60;
        landmarks = result.landmarks && result.landmarks[0] ? result.landmarks[0] : [];
        this._handle(landmarks, frame, latencyMs);
      }
    } else if (this.mode === "demo") {
      // 示範模式沒有真的影格，直接用鏡頭區塊的寬高，骨架才不會被裁掉
      frame = { width: this.canvas.clientWidth || 640, height: this.canvas.clientHeight || 480 };
      landmarks = demoLandmarks(now / 1000, frame.width / frame.height);
      this._handle(landmarks, frame, null);
    }

    requestAnimationFrame(this._loop);
  }

  _handle(landmarks, frame, latencyMs) {
    this._draw(landmarks, frame);
    const visible = landmarks.length > 0 && vis(landmarks[11]) && vis(landmarks[12]);
    if (this.mode === "camera") {
      this.onStatus(visible ? { text: "33 點追蹤中", kind: "ok", live: true } : { text: "請站回畫面", kind: "warn", live: true });
    }
    // 人離開畫面：不送新姿勢，人偶停在最後一個穩定姿勢（計畫書 2.2）
    if (!visible) return;
    this.onPose(toAngles(landmarks, frame), { latencyMs, source: this.mode });
  }

  _draw(landmarks, frame) {
    const { canvas, ctx } = this;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    if (canvas.width !== Math.round(w * dpr) || canvas.height !== Math.round(h * dpr)) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    // 跟 video 的 object-fit: cover 一樣的換算，關鍵點才會疊在人身上
    const scale = Math.max(w / frame.width, h / frame.height);
    const ox = (w - frame.width * scale) / 2;
    const oy = (h - frame.height * scale) / 2;
    const px = (p) => [ox + p.x * frame.width * scale, oy + p.y * frame.height * scale];

    if (this.mode === "demo") drawDemoSilhouette(ctx, landmarks, px, w, h);
    if (!this.showSkeleton || !landmarks.length) return;

    ctx.lineWidth = 3;
    ctx.strokeStyle = "#3ccf8e";
    ctx.lineCap = "round";
    for (const [a, b] of BONES) {
      if (!vis(landmarks[a]) || !vis(landmarks[b])) continue;
      const [x1, y1] = px(landmarks[a]);
      const [x2, y2] = px(landmarks[b]);
      ctx.beginPath();
      ctx.moveTo(x1, y1);
      ctx.lineTo(x2, y2);
      ctx.stroke();
    }
    ctx.fillStyle = "#ffffff";
    for (const i of KEY_POINTS) {
      if (!vis(landmarks[i])) continue;
      const [x, y] = px(landmarks[i]);
      ctx.beginPath();
      ctx.arc(x, y, 3.5, 0, Math.PI * 2);
      ctx.fill();
    }
  }
}

function vis(p) {
  return p && (p.visibility === undefined || p.visibility > 0.5);
}

// 關鍵點 → 人偶角度。關鍵點是鏡頭原始座標，先鏡像成「螢幕座標」再算。
// 人的右手（12/14/16）鏡像後出現在螢幕右邊。
function toAngles(lm, frame) {
  const aspect = frame.width / frame.height;
  const s = (i) => ({ x: (1 - lm[i].x) * aspect, y: lm[i].y, ok: vis(lm[i]) });

  const armAngles = (shoulder, elbow, wrist) => {
    const S = s(shoulder), E = s(elbow), W = s(wrist);
    if (!E.ok) return null;
    const upper = Math.atan2(E.x - S.x, E.y - S.y);
    const fore = W.ok ? Math.atan2(W.x - E.x, W.y - E.y) : upper;
    return { upper, fore };
  };

  const ls = s(11), rs = s(12);
  const shoulderMid = { x: (ls.x + rs.x) / 2, y: (ls.y + rs.y) / 2 };

  let lean = 0;
  if (vis(lm[23]) && vis(lm[24])) {
    const lh = s(23), rh = s(24);
    const hipMid = { x: (lh.x + rh.x) / 2, y: (lh.y + rh.y) / 2 };
    lean = Math.atan2(shoulderMid.x - hipMid.x, hipMid.y - shoulderMid.y);
  }

  let head = 0;
  if (vis(lm[0])) {
    const n = s(0);
    head = Math.atan2(n.x - shoulderMid.x, shoulderMid.y - n.y) - lean;
  }

  return {
    right: armAngles(12, 14, 16),
    left: armAngles(11, 13, 15),
    lean: clamp(lean, -0.5, 0.5),
    head: clamp(head, -0.6, 0.6),
  };
}

// 示範動作：用簡單的 2D 正向運動學產生 33 點中會用到的幾個點（螢幕座標），
// 再轉回鏡頭原始座標（x 鏡像），讓示範與真鏡頭走同一條計算路徑。
function demoLandmarks(t, aspect) {
  const wave = 0.5 + 0.5 * Math.sin(t * 1.1);
  const rightUpper = 0.25 + 1.9 * wave;
  const rightFore = rightUpper + 0.35 * Math.sin(t * 4) * wave + 0.2;
  const leftUpper = -(0.2 + 0.45 * (0.5 + 0.5 * Math.sin(t * 0.7 + 1)));
  const leftFore = leftUpper - 0.35;
  const lean = 0.07 * Math.sin(t * 0.55);

  // 手臂水平伸直時半寬約 0.42，窄的區塊要整體縮小才放得下
  const k = Math.min(1, (aspect / 2) / 0.44);
  const U = 0.15 * k, F = 0.13 * k, TORSO = 0.27 * k, SH = 0.09 * k, HIP = 0.06 * k;
  const hipMid = { x: 0.5 * aspect, y: 0.6 };
  const sm = { x: hipMid.x + Math.sin(lean) * TORSO, y: hipMid.y - Math.cos(lean) * TORSO };
  const perp = { x: Math.cos(lean), y: Math.sin(lean) };

  const pts = {};
  pts[11] = { x: sm.x - perp.x * SH, y: sm.y - perp.y * SH };
  pts[12] = { x: sm.x + perp.x * SH, y: sm.y + perp.y * SH };
  const limb = (from, angle, len) => ({ x: from.x + Math.sin(angle) * len, y: from.y + Math.cos(angle) * len });
  pts[13] = limb(pts[11], leftUpper, U);
  pts[15] = limb(pts[13], leftFore, F);
  pts[14] = limb(pts[12], rightUpper, U);
  pts[16] = limb(pts[14], rightFore, F);
  pts[0] = { x: sm.x + Math.sin(lean + 0.1 * Math.sin(t * 0.9)) * 0.11 * k, y: sm.y - Math.cos(lean) * 0.11 * k };
  pts[23] = { x: hipMid.x - HIP, y: hipMid.y };
  pts[24] = { x: hipMid.x + HIP, y: hipMid.y };
  pts[25] = { x: pts[23].x - 0.01 * k, y: hipMid.y + 0.17 * k };
  pts[26] = { x: pts[24].x + 0.01 * k, y: hipMid.y + 0.17 * k };
  pts[27] = { x: pts[25].x, y: hipMid.y + 0.33 * k };
  pts[28] = { x: pts[26].x, y: hipMid.y + 0.33 * k };

  const out = new Array(33).fill(null).map(() => ({ x: 0, y: 0, visibility: 0 }));
  for (const [i, p] of Object.entries(pts)) {
    out[i] = { x: 1 - p.x / aspect, y: p.y, visibility: 1 };
  }
  return out;
}

function drawDemoSilhouette(ctx, lm, px, w, h) {
  if (!lm.length) return;
  const [hx, hy] = px(lm[0]);
  const [lx, ly] = px(lm[11]);
  const [rx, ry] = px(lm[12]);
  const r = Math.abs(rx - lx) * 0.42;
  ctx.fillStyle = "rgba(255,255,255,0.08)";
  ctx.beginPath();
  ctx.arc(hx, hy - r * 0.2, r, 0, Math.PI * 2);
  ctx.fill();
  // 小窗（手機）放不下提示文字就不畫
  if (w < 300) return;
  ctx.font = "13px system-ui, sans-serif";
  ctx.fillStyle = "rgba(255,255,255,0.45)";
  ctx.save();
  // canvas 整個被 CSS 鏡像了，文字要再翻回來
  ctx.scale(-1, 1);
  ctx.textAlign = "center";
  ctx.fillText("示範動作：按「開啟鏡頭」換成你自己", -w / 2, h - 44);
  ctx.restore();
}

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}
