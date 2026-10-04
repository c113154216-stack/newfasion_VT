// 3D 網格人偶＋衣服（Three.js）。
// 現在的人偶是用基本幾何組出來的替身，衣服也是依尺寸表算出來的簡化外形；
// 之後換成紀泓宇的 SMPL-X 本人模型，以及我們生成的衣服網格（GLB）。

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

const BASE_HEIGHT = 175;
// 胸圍是橢圓：寬比深大，用這兩個比例把「等效圓半徑」拉成橢圓
const WIDTH_K = 1.18;
const DEPTH_K = 0.8;

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(28, 1, 0.05, 50);
    this.camera.position.set(0, 1.25, 4.2);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.target.set(0, 1.05, 0);
    this.controls.enableDamping = true;
    this.controls.minDistance = 1.6;
    this.controls.maxDistance = 7;
    this.controls.maxPolarAngle = Math.PI * 0.55;
    this.controls.update();

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8478, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.6);
    key.position.set(1.5, 3, 2.5);
    this.scene.add(key);
    const rim = new THREE.DirectionalLight(0xffffff, 0.6);
    rim.position.set(-2, 2, -2);
    this.scene.add(rim);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(0.9, 48),
      new THREE.MeshBasicMaterial({ color: 0x000000, transparent: true, opacity: 0.07 }),
    );
    floor.rotation.x = -Math.PI / 2;
    this.scene.add(floor);

    this.skinMat = new THREE.MeshStandardMaterial({ color: 0xa9a49c, roughness: 0.75, flatShading: true });
    this.wireMat = new THREE.MeshBasicMaterial({ color: 0x000000, wireframe: true, transparent: true, opacity: 0.06 });

    this.body = null;
    this.product = null;
    this.size = null;
    this.simMode = "xpbd";
    this.pose = { left: { upper: -0.25, fore: -0.3 }, right: { upper: 0.25, fore: 0.3 }, lean: 0, head: 0 };
    this.target = structuredClone(this.pose);
    this.hem = { angle: 0, vel: 0 };
    this.prevLean = 0;
    this.fps = 0;
    this._frames = 0;
    this._fpsT = performance.now();
    this._last = performance.now();

    this.rig = null;
    this.garment = null;

    new ResizeObserver(() => this._resize()).observe(container);
    this._resize();
    this._tick = this._tick.bind(this);
    requestAnimationFrame(this._tick);
  }

  setBody(body) {
    this.body = body;
    this._buildRig();
    this._buildGarment();
  }

  setGarment(product, size) {
    this.product = product;
    this.size = size;
    this._buildGarment();
  }

  setSimMode(mode) {
    this.simMode = mode;
    if (mode === "none") this.hem = { angle: 0, vel: 0 };
  }

  setPose(angles) {
    // 偵測不到某隻手時保留上一個目標角度
    if (angles.left) this.target.left = angles.left;
    if (angles.right) this.target.right = angles.right;
    this.target.lean = angles.lean;
    this.target.head = angles.head;
  }

  // 初始穿入估計：衣服胸圍半徑小於身體胸圍半徑就會陷進去。
  // 正式版要用衣服頂點對身體網格算有號距離，這裡只看胸圍一圈。
  get collisionEstimate() {
    if (!this.body || !this.garmentChest) return null;
    const gap = this.garmentChest - this.body.chest;
    if (gap >= 2) return 0;
    return Math.min(100, Math.round(((2 - gap) / 10) * 100 * 10) / 10);
  }

  _resize() {
    if (!this.controls) return;
    const w = this.container.clientWidth || 1;
    const h = this.container.clientHeight || 1;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // 直式畫面（手機）把相機拉遠，整個人才放得進來
    const dist = 4.2 * Math.max(1, 0.75 / this.camera.aspect);
    const dir = this.camera.position.clone().sub(this.controls.target).normalize();
    this.camera.position.copy(this.controls.target).addScaledVector(dir, dist);
    this.camera.updateProjectionMatrix();
  }

  _buildRig() {
    if (this.rig) {
      this.scene.remove(this.rig.root);
      disposeTree(this.rig.root);
    }
    const b = this.body;
    const s = b.height / BASE_HEIGHT;
    const rChest = b.chest / (2 * Math.PI) / 100;
    const rWaist = b.waist / (2 * Math.PI) / 100;
    const rHip = b.hip / (2 * Math.PI) / 100;
    const torsoH = 0.5 * s;
    const hipY = 0.93 * s;

    const root = new THREE.Group();
    const pelvis = new THREE.Group();
    pelvis.position.y = hipY;
    root.add(pelvis);

    // 軀幹：用旋轉體（lathe）依臀、腰、胸圍做出輪廓，再壓成橢圓截面
    const torsoProfile = [
      [rHip * 0.55, -0.06 * s],
      [rHip, 0.02 * s],
      [rWaist, 0.2 * s],
      [rChest, 0.34 * s],
      [rChest * 0.93, 0.44 * s],
      [0.06, torsoH],
    ].map(([r, y]) => new THREE.Vector2(r, y));
    const spine = new THREE.Group();
    pelvis.add(spine);
    const torso = new THREE.Mesh(new THREE.LatheGeometry(torsoProfile, 28), this.skinMat);
    torso.scale.set(WIDTH_K, 1, DEPTH_K);
    spine.add(torso);
    torso.add(new THREE.Mesh(torso.geometry, this.wireMat));

    const neck = new THREE.Group();
    neck.position.y = torsoH;
    spine.add(neck);
    const neckMesh = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.055, 0.1 * s, 12), this.skinMat);
    neckMesh.position.y = 0.04 * s;
    neck.add(neckMesh);
    const head = new THREE.Mesh(new THREE.SphereGeometry(0.1 * s, 20, 16), this.skinMat);
    head.scale.set(0.9, 1.15, 1);
    head.position.y = 0.19 * s;
    neck.add(head);

    // 手臂：肩關節 → 上臂 → 手肘 → 前臂。肩寬由身體量測值決定
    const shoulderX = Math.max(0.12, b.shoulder / 200 - 0.035);
    const shoulderY = 0.42 * s;
    const upperLen = 0.29 * s;
    const foreLen = 0.25 * s;
    const makeArm = (side) => {
      const shoulder = new THREE.Group();
      shoulder.position.set(side * shoulderX, shoulderY, 0);
      spine.add(shoulder);
      const upper = new THREE.Mesh(new THREE.CapsuleGeometry(0.043, upperLen - 0.08, 4, 10), this.skinMat);
      upper.position.y = -upperLen / 2;
      shoulder.add(upper);
      const elbow = new THREE.Group();
      elbow.position.y = -upperLen;
      shoulder.add(elbow);
      const fore = new THREE.Mesh(new THREE.CapsuleGeometry(0.036, foreLen - 0.07, 4, 10), this.skinMat);
      fore.position.y = -foreLen / 2;
      elbow.add(fore);
      const hand = new THREE.Mesh(new THREE.SphereGeometry(0.045, 12, 10), this.skinMat);
      hand.scale.set(0.8, 1.2, 0.5);
      hand.position.y = -foreLen - 0.03;
      elbow.add(hand);
      return { shoulder, elbow, upperLen, foreLen };
    };
    // three.js 的 +x 是螢幕右邊（相機在 +z 往 -z 看）
    const armR = makeArm(+1);
    const armL = makeArm(-1);

    const makeLeg = (side) => {
      const hip = new THREE.Group();
      hip.position.set(side * rHip * 0.55, -0.02 * s, 0);
      pelvis.add(hip);
      const thigh = new THREE.Mesh(new THREE.CapsuleGeometry(0.065, 0.36 * s, 4, 10), this.skinMat);
      thigh.position.y = -0.22 * s;
      hip.add(thigh);
      const shin = new THREE.Mesh(new THREE.CapsuleGeometry(0.048, 0.36 * s, 4, 10), this.skinMat);
      shin.position.y = -0.66 * s;
      hip.add(shin);
      const foot = new THREE.Mesh(new THREE.BoxGeometry(0.09, 0.05, 0.22), this.skinMat);
      foot.position.set(0, -hipY + 0.045, 0.05);
      hip.add(foot);
    };
    makeLeg(+1);
    makeLeg(-1);

    this.scene.add(root);
    this.rig = { root, spine, neck, armL, armR, shoulderY, shoulderX, torsoH, s, rChest };
  }

  _buildGarment() {
    if (!this.rig) return;
    if (this.garment) {
      for (const g of this.garment.parts) {
        g.parent && g.parent.remove(g);
        disposeTree(g);
      }
      this.garment = null;
    }
    this.garmentChest = null;
    if (!this.product) return;

    const p = this.product;
    const b = this.body;
    // 沒有尺寸表的上傳衣服：只做上身預覽，用身體＋6 cm 生成
    const dims = p.sizeChart && this.size
      ? p.sizeChart[this.size]
      : { chest: b.chest + 6, shoulder: b.shoulder + 1, length: b.height * 0.39 };
    this.garmentChest = dims.chest;

    const { spine, armL, armR, shoulderY, s } = this.rig;
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(p.color || "#8a8a8a"),
      roughness: 0.92,
      side: THREE.DoubleSide,
    });
    const rG = dims.chest / (2 * Math.PI) / 100;
    const rHem = p.fit === "loose" ? rG * 1.02 : rG * 0.96;
    const len = dims.length / 100;
    const top = shoulderY + 0.07 * s;
    const split = len * 0.5;

    // 衣身上半（跟著軀幹）
    const upperProfile = [
      [rG * 0.99, -split],
      [rG, -len * 0.3],
      [rG * 0.97, -0.08],
      [Math.max(0.09, dims.shoulder / 200 - 0.06), -0.02],
      [0.075, 0],
    ].map(([r, y]) => new THREE.Vector2(r, y));
    const upper = new THREE.Mesh(new THREE.LatheGeometry(upperProfile, 32), mat);
    upper.scale.set(WIDTH_K, 1, DEPTH_K);
    upper.position.y = top;
    spine.add(upper);

    // 衣身下半（下擺）：XPBD 模式下會延遲擺動，示意布料的慣性
    const hemPivot = new THREE.Group();
    hemPivot.position.y = top - split;
    spine.add(hemPivot);
    const lowerProfile = [
      [rHem, -(len - split)],
      [(rHem + rG) / 2, -(len - split) * 0.5],
      [rG * 0.99, 0],
    ].map(([r, y]) => new THREE.Vector2(r, y));
    const lower = new THREE.Mesh(new THREE.LatheGeometry(lowerProfile, 32), mat);
    lower.scale.set(WIDTH_K, 1, DEPTH_K);
    hemPivot.add(lower);

    // 袖子：短袖只包上臂，長袖再加前臂
    const sleeves = [];
    for (const arm of [armL, armR]) {
      const cap = new THREE.Mesh(new THREE.SphereGeometry(0.068, 16, 12, 0, Math.PI * 2, 0, Math.PI / 2), mat);
      cap.position.y = 0.005;
      arm.shoulder.add(cap);
      const shortLen = Math.min(arm.upperLen * 0.62, 0.19);
      const upperSleeveLen = p.sleeve === "long" ? arm.upperLen + 0.02 : shortLen;
      const sleeve = new THREE.Mesh(
        new THREE.CylinderGeometry(0.068, p.sleeve === "long" ? 0.056 : 0.062, upperSleeveLen, 16, 1, true),
        mat,
      );
      sleeve.position.y = -upperSleeveLen / 2;
      arm.shoulder.add(sleeve);
      sleeves.push(cap, sleeve);
      if (p.sleeve === "long") {
        const foreSleeve = new THREE.Mesh(
          new THREE.CylinderGeometry(0.056, 0.048, arm.foreLen - 0.02, 16, 1, true),
          mat,
        );
        foreSleeve.position.y = -(arm.foreLen - 0.02) / 2;
        arm.elbow.add(foreSleeve);
        sleeves.push(foreSleeve);
      }
    }

    this.garment = { parts: [upper, hemPivot, ...sleeves], hemPivot, mat };
  }

  _tick(now) {
    const dt = Math.min(0.05, (now - this._last) / 1000);
    this._last = now;

    this._frames++;
    if (now - this._fpsT >= 500) {
      this.fps = (this._frames * 1000) / (now - this._fpsT);
      this._frames = 0;
      this._fpsT = now;
    }

    if (this.rig) this._applyPose(dt);
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
    requestAnimationFrame(this._tick);
  }

  _applyPose(dt) {
    const k = Math.min(1, dt * 14);
    const cur = this.pose;
    const tgt = this.target;
    for (const side of ["left", "right"]) {
      cur[side].upper += (tgt[side].upper - cur[side].upper) * k;
      cur[side].fore += (tgt[side].fore - cur[side].fore) * k;
    }
    cur.lean += (tgt.lean - cur.lean) * k;
    cur.head += (tgt.head - cur.head) * k;

    const { spine, neck, armL, armR } = this.rig;
    // 角度定義見 pose.js；spine 往右傾要繞 z 軸轉負角度，子物件的手臂要扣回去
    spine.rotation.z = -cur.lean;
    neck.rotation.z = -cur.head * 0.6;
    armR.shoulder.rotation.z = cur.right.upper + cur.lean;
    armR.elbow.rotation.z = cur.right.fore - cur.right.upper;
    armL.shoulder.rotation.z = cur.left.upper + cur.lean;
    armL.elbow.rotation.z = cur.left.fore - cur.left.upper;

    // 下擺擺動：簡單的彈簧阻尼，由軀幹轉動速度帶動（物理還沒接上前的示意）
    if (this.garment) {
      if (this.simMode === "none") {
        this.garment.hemPivot.rotation.z = 0;
      } else {
        const leanVel = (cur.lean - this.prevLean) / Math.max(dt, 1e-3);
        const acc = -90 * this.hem.angle - 9 * this.hem.vel + leanVel * 6;
        this.hem.vel += acc * dt;
        this.hem.angle = Math.max(-0.25, Math.min(0.25, this.hem.angle + this.hem.vel * dt));
        this.garment.hemPivot.rotation.z = this.hem.angle;
      }
    }
    this.prevLean = cur.lean;
  }
}

function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
  });
}
