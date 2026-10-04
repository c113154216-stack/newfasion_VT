import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

let renderer;
let scene;
let camera;
let controls;
let currentRoot;
const loader = new GLTFLoader();

function ensureViewer(host) {
  if (renderer) return;

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x090807);

  camera = new THREE.PerspectiveCamera(45, 1, 0.01, 200);
  camera.position.set(1.6, 1.2, 2.2);

  renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  host.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.target.set(0, 0.8, 0);

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xfff2dd, 1.1);
  key.position.set(2.4, 4, 2);
  scene.add(key);
  const fill = new THREE.DirectionalLight(0x88a0b8, 0.45);
  fill.position.set(-2, 1.5, -1.5);
  scene.add(fill);

  scene.add(new THREE.GridHelper(4, 12, 0x3a342c, 0x221e1a));

  const resize = () => {
    const w = host.clientWidth || 640;
    const h = host.clientHeight || 520;
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
    renderer.setSize(w, h, false);
  };
  new ResizeObserver(resize).observe(host);
  resize();

  const tick = () => {
    requestAnimationFrame(tick);
    if (controls) controls.update();
    renderer.render(scene, camera);
  };
  tick();
}

function makePlaceholder() {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(0.7, 1.2, 0.4),
    new THREE.MeshStandardMaterial({ color: 0xc4a574, roughness: 0.55, metalness: 0.05 }),
  );
  mesh.position.y = 0.6;
  mesh.name = "TryOnPlaceholder";
  return mesh;
}

function countMeshes(root) {
  let n = 0;
  root.traverse((child) => {
    if (child.isMesh) n += 1;
  });
  return n;
}

function fitObject(object) {
  if (!object || !camera || !controls) return;
  try {
    object.updateWorldMatrix(true, true);
    const box = new THREE.Box3().setFromObject(object);
    if (box.isEmpty() || !Number.isFinite(box.min.x)) {
      throw new Error("empty bounds");
    }
    const size = box.getSize(new THREE.Vector3());
    const center = box.getCenter(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z, 0.001);
    const dist = maxDim * 2.2;
    camera.position.set(center.x + dist * 0.6, center.y + dist * 0.35, center.z + dist);
    controls.target.copy(center);
    controls.update();
  } catch (_err) {
    camera.position.set(1.6, 1.2, 2.2);
    controls.target.set(0, 0.6, 0);
    controls.update();
  }
}

function disposeRoot(root) {
  if (!root) return;
  scene.remove(root);
  root.traverse((child) => {
    if (child.geometry) child.geometry.dispose();
    if (child.material) {
      const mats = Array.isArray(child.material) ? child.material : [child.material];
      for (const mat of mats) mat.dispose();
    }
  });
}

export function initTryOnViewer(host) {
  if (!host) return;
  ensureViewer(host);
}

export async function loadTryOnGlb(url) {
  if (!renderer) {
    throw new Error("試穿畫布尚未初始化");
  }
  disposeRoot(currentRoot);
  currentRoot = null;

  let root;
  try {
    const gltf = await loader.loadAsync(url);
    root = gltf.scene;
  } catch (err) {
    throw new Error(err && err.message ? `模型載入失敗：${err.message}` : "模型載入失敗");
  }

  if (!root || !root.isObject3D) {
    root = makePlaceholder();
  } else if (countMeshes(root) === 0) {
    root.add(makePlaceholder());
  }

  currentRoot = root;
  scene.add(currentRoot);
  fitObject(currentRoot);
}
