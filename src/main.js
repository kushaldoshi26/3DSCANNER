import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';
import {
  createBearingAssembly,
  createNutHolderNuts,
  createCameraSensor,
  createBaseHardware,
} from './hardware-models.js';

gsap.registerPlugin(ScrollTrigger);

const $ = (s) => document.querySelector(s);
const chapters = [...document.querySelectorAll('.chapter')];
const navLinks = [...document.querySelectorAll('[data-chapter]')];
const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)');
const mobileQuery = matchMedia('(max-width: 700px)');
const modelUrl = (name) => `${import.meta.env.BASE_URL}models/${name}`;
const clamp = THREE.MathUtils.clamp;
const lerp = THREE.MathUtils.lerp;
const smooth = (n) => n * n * (3 - 2 * n);
const halfPi = Math.PI / 2;
const sceneElement = $('#scene');
const canvas = $('#webgl');
const loaderElement = $('#loader');
const controls = $('#motion-toggle');
let paused = reduceMotion.matches;
let ready = false;
let activeChapter = -1;
let scrollState = { position: 0 };
let anchors = [];
let renderRequested = true;
let failed = false;
// Curated distinctive color palette for each component
const PARTS = {
  frame: { label: 'MAIN FRAME', number: '01', color: '#445761', roughness: .34, metalness: .32, position: [0, -1.4, 0], rotation: [halfPi, 0, 0] },
  top_plate: { label: 'TOP PLATE', number: '02', color: '#ded5c3', roughness: .28, metalness: .30, position: [0, -.93, 0], rotation: [0, 0, 0] },
  small_gear: { label: 'SMALL GEAR', number: '03', color: '#d97b2b', roughness: .26, metalness: .48, position: [.7, -1.3, -.5], rotation: [0, 0, 0] },
  nut_holder: { label: 'NUT HOLDER', number: '04', color: '#e5ded2', roughness: .32, metalness: .25, position: [-.4, -1.25, .9], rotation: [halfPi, 0, 0] },
  bearing_holder: { label: 'BEARING HOLDER', number: '05', color: '#b88344', roughness: .29, metalness: .42, position: [0, -1.16, 0], rotation: [halfPi, 0, 0] },
  camera_stand: { label: 'CAMERA STAND', number: '06', color: '#4e626c', roughness: .36, metalness: .32, position: [2.25, .873, 0], rotation: [0, 0, 0] },
};

// Placements are a visual assembly, not validated mechanical CAD constraints.
// Coordinates use the original STL numerical units at a common 0.023 scale.
function partPose(name, overrides = {}) {
  return { p: PARTS[name].position, r: PARTS[name].rotation, s: 1, a: 1, ...overrides };
}
const assembledParts = () => Object.fromEntries(Object.keys(PARTS).map((id) => [id, partPose(id)]));
const SHOTS = [
  { x: .22, y: -.03, scale: 1.02, ry: -.38, rx: -.08, look: [.6, .7, 0], parts: assembledParts() },
  { x: .22, y: -.03, scale: 1.12, ry: .3, rx: .1, look: [.5, .2, 0], parts: {
    frame: partPose('frame', { p: [0, -1.5, 0] }),
    top_plate: partPose('top_plate', { p: [0, 1.2, 0], r: [-1.08, 0, 0] }),
    small_gear: partPose('small_gear', { p: [.7, -.25, -.3] }),
    nut_holder: partPose('nut_holder', { p: [-.4, -.5, .85] }),
    bearing_holder: partPose('bearing_holder', { p: [0, .1, 0] }),
    camera_stand: partPose('camera_stand', { p: [3.1, .85, -1], a: .04 }),
  } },
  { x: -.22, y: -.03, scale: 1.25, ry: -.2, rx: .3, look: [0, 0, 0], parts: {
    frame: partPose('frame', { p: [0, -2, -.8], a: .12, s: .85 }),
    top_plate: partPose('top_plate', { p: [-1.6, 2.1, -1], a: .09, s: .7 }),
    small_gear: partPose('small_gear', { p: [-.7, 1, .2], s: 3.4, r: [.55, .5, .08] }),
    nut_holder: partPose('nut_holder', { p: [.95, -.5, .65], s: 3.2, r: [.15, -.2, -.1] }),
    bearing_holder: partPose('bearing_holder', { p: [-.15, -.3, -.3], s: 1.7, r: [.05, -.3, .1] }),
    camera_stand: partPose('camera_stand', { p: [4, .85, -1.8], a: 0 }),
  } },
  { x: .23, y: -.02, scale: 1.08, ry: -.55, rx: 0, look: [.35, .4, 0], parts: {
    frame: partPose('frame', { p: [-1.5, -2.3, -1.5], a: .1, s: .7 }),
    top_plate: partPose('top_plate', { p: [-1.5, -1.7, -1.5], a: .14, s: .7 }),
    small_gear: partPose('small_gear', { p: [-1.5, -1.95, -1.5], a: 0 }),
    nut_holder: partPose('nut_holder', { p: [-1.5, -1.95, -1.5], a: 0 }),
    bearing_holder: partPose('bearing_holder', { p: [-1.5, -1.95, -1.5], a: 0 }),
    camera_stand: partPose('camera_stand', { p: [.7, .3, 0], s: 1.08, r: [0, .5, 0] }),
  } },
  { x: .2, y: -.02, scale: .92, ry: .25, rx: .06, look: [.7, .35, 0], parts: {
    frame: partPose('frame', { p: [0, -1.85, 0] }),
    top_plate: partPose('top_plate', { p: [0, 2.2, 0], r: [.08, 0, 0] }),
    small_gear: partPose('small_gear', { p: [-.9, .55, .4], s: 1.5 }),
    nut_holder: partPose('nut_holder', { p: [1, .35, .7], s: 1.5 }),
    bearing_holder: partPose('bearing_holder', { p: [0, -.45, 0] }),
    camera_stand: partPose('camera_stand', { p: [3, .5, -.3] }),
  } },
  { x: .14, y: 0, scale: 1.04, ry: -.45, rx: -.04, look: [.6, .7, 0], parts: assembledParts() },
];

function updateMotionButton() {
  controls.disabled = reduceMotion.matches || failed;
  if (reduceMotion.matches) {
    controls.setAttribute('aria-pressed', 'true');
    controls.setAttribute('aria-label', 'Ambient rotation disabled by your reduced motion preference');
    controls.title = 'Reduced motion is enabled on your device';
    return;
  }
  controls.setAttribute('aria-pressed', String(paused));
  controls.setAttribute('aria-label', paused ? 'Resume ambient rotation' : 'Pause ambient rotation');
  controls.title = paused ? 'Resume ambient rotation' : 'Pause ambient rotation';
}
updateMotionButton();
controls.addEventListener('click', () => { paused = !paused; updateMotionButton(); renderRequested = true; });
$('#reset-view').addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: reduceMotion.matches ? 'instant' : 'smooth' });
  renderRequested = true;
});

function updateAnchors() {
  const vh = innerHeight;
  anchors = chapters.map((el, i) => i === 0 ? 0 : el.offsetTop + (i < 4 ? Math.max(0, (el.offsetHeight - vh) / 2) : 0));
  anchors.push($('#specifications').offsetTop - vh * 1.08);
  renderRequested = true;
}
updateAnchors();

function activateChapter(index) {
  if (index === activeChapter) return;
  activeChapter = index;
  chapters.forEach((el, i) => el.classList.toggle('is-active', i === index));
  navLinks.forEach((el, i) => {
    el.classList.toggle('is-active', i === index);
    if (i === index) el.setAttribute('aria-current', 'step'); else el.removeAttribute('aria-current');
  });
  const labels = ['01 / ASSEMBLY', '02 / STRUCTURE', '03 / MECHANISM', '04 / PERSPECTIVE', '05 / EXPLODED VIEW'];
  $('#scene-status').textContent = labels[index];
}

// One scrubbed timeline is the source of truth in either scroll direction.
const scrollTween = gsap.fromTo(scrollState, { position: 0 }, {
  position: 1, duration: 1, ease: 'none',
  onUpdate: () => { renderRequested = true; },
  scrollTrigger: {
    trigger: '#story', start: 'top top', end: 'bottom center',
    scrub: reduceMotion.matches ? true : .7,
    invalidateOnRefresh: true,
    onRefresh: updateAnchors,
  },
});
ScrollTrigger.create({
  trigger: '#specifications', start: 'top 45%',
  onEnter: () => { document.body.classList.add('in-specifications'); },
  onLeaveBack: () => { document.body.classList.remove('in-specifications'); renderRequested = true; },
});

let renderer, scene, camera, assembly, floor, ring, shadow, pmremTarget;
const meshes = {};
const materials = {};
const labels = {};
const target = new THREE.Vector3();
const world = new THREE.Vector3();
let raf = 0;
let previousTime = 0;
let ambientTime = 0;
let visible = !document.hidden;

function showFallback(message) {
  failed = true;
  ready = false;
  loaderElement.classList.add('is-hidden');
  canvas.style.display = 'none';
  const poster = document.createElement('img');
  poster.src = modelUrl('assembly-poster.png');
  poster.alt = 'The scanner frame, top plate and camera stand assembled in a studio view';
  poster.className = 'assembly-poster';
  sceneElement.appendChild(poster);
  $('#scene-status').textContent = message;
  controls.disabled = true;
  Object.values(labels).forEach((label) => label.element.remove());
}

function createLabel(id) {
  const element = document.createElement('div');
  element.className = 'part-label';
  element.innerHTML = `<span class="part-label-dot"></span><span class="part-label-line"></span><span class="part-label-text"><span>${PARTS[id].number}</span> ${PARTS[id].label}</span>`;
  element.setAttribute('aria-hidden', 'true');
  sceneElement.appendChild(element);
  labels[id] = { element };
}

function setupScene() {
  renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true, powerPreference: 'high-performance' });
  renderer.setPixelRatio(Math.min(devicePixelRatio, mobileQuery.matches ? 1.5 : 2));
  renderer.setSize(innerWidth, innerHeight);
  renderer.setClearColor(0x102630, 0);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.05;
  renderer.shadowMap.enabled = !mobileQuery.matches;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(32, innerWidth / innerHeight, .1, 100);
  camera.position.set(7, 5.5, 11);
  camera.lookAt(.5, .4, 0);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = new RoomEnvironment();
  pmremTarget = pmrem.fromScene(environment, .04);
  scene.environment = pmremTarget.texture;
  scene.environmentIntensity = .65;
  environment.dispose();
  pmrem.dispose();
  scene.add(new THREE.HemisphereLight('#e9e8e0', '#182f39', .65));
  const key = new THREE.DirectionalLight('#fff0dc', 2.4);
  key.position.set(-4, 7, 6);
  key.castShadow = true;
  key.shadow.mapSize.set(2048, 2048);
  Object.assign(key.shadow.camera, { left: -8, right: 8, top: 8, bottom: -8, near: .1, far: 30 });
  key.shadow.normalBias = .018;
  key.shadow.bias = -.0001;
  key.shadow.radius = 3;
  scene.add(key);
  const rim = new THREE.DirectionalLight('#b9d7eb', 2.4);
  rim.position.set(5, 5, -5); scene.add(rim);
  const fill = new THREE.DirectionalLight('#cda984', .65);
  fill.position.set(4, 0, 7); scene.add(fill);
  assembly = new THREE.Group();
  scene.add(assembly);
  floor = new THREE.Group(); assembly.add(floor);
  floor.position.y = -1.87;
  const points = [];
  for (let i = 0; i <= 160; i++) { const angle = i / 160 * Math.PI * 2; points.push(new THREE.Vector3(Math.cos(angle) * 2.05, .01, Math.sin(angle) * 2.05)); }
  ring = new THREE.Line(new THREE.BufferGeometry().setFromPoints(points), new THREE.LineBasicMaterial({ color: '#9b8c79', transparent: true, opacity: .18 }));
  floor.add(ring);
  const shadowCanvas = document.createElement('canvas'); shadowCanvas.width = shadowCanvas.height = 128;
  const ctx = shadowCanvas.getContext('2d');
  const gradient = ctx.createRadialGradient(64, 64, 12, 64, 64, 64);
  gradient.addColorStop(0, 'rgba(0,0,0,.6)'); gradient.addColorStop(.5, 'rgba(0,0,0,.28)'); gradient.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, 128, 128);
  shadow = new THREE.Mesh(new THREE.PlaneGeometry(6, 6), new THREE.MeshBasicMaterial({ map: new THREE.CanvasTexture(shadowCanvas), transparent: true, opacity: .5, depthWrite: false }));
  shadow.rotation.x = -halfPi; floor.add(shadow);
  canvas.addEventListener('webglcontextlost', (event) => { event.preventDefault(); cancelAnimationFrame(raf); if (!failed) showFallback('STATIC PREVIEW'); });
}

async function loadModels() {
  const loader = new GLTFLoader();
  let loaded = 0;
  await Promise.all(Object.entries(PARTS).map(async ([id, part]) => {
    const gltf = await loader.loadAsync(modelUrl(`${id}.glb`));
    const model = gltf.scene;
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);
    const material = new THREE.MeshStandardMaterial({ color: part.color, roughness: part.roughness, metalness: part.metalness, envMapIntensity: 1.1 });
    model.traverse((node) => { if (node.isMesh) { node.material.dispose?.(); node.material = material; node.castShadow = true; node.receiveShadow = true; } });
    const normalized = new THREE.Group(); normalized.add(model); normalized.scale.setScalar(.023);
    const pivot = new THREE.Group(); pivot.add(normalized);

    // Attach authentic hardware accessories to corresponding printed parts
    if (id === 'bearing_holder') {
      const bearingAssembly = createBearingAssembly();
      pivot.add(bearingAssembly);
    } else if (id === 'nut_holder') {
      const nutAssembly = createNutHolderNuts();
      pivot.add(nutAssembly);
    } else if (id === 'camera_stand') {
      const cameraSensor = createCameraSensor();
      pivot.add(cameraSensor);
    }

    assembly.add(pivot); meshes[id] = pivot; materials[id] = material;
    createLabel(id);
    loaded++;
    $('#load-progress').style.width = `${loaded / 6 * 100}%`;
    $('#load-label').textContent = `${String(loaded).padStart(2, '0')} / 06 PARTS READY`;
  }));
  if (failed) return;
  ready = true;
  renderRequested = true;
  applyScene(0);
  // Compile once before removing the loader to avoid a blank first frame.
  await renderer.compileAsync(scene, camera);
  renderer.render(scene, camera);
  loaderElement.classList.add('is-hidden');
  document.body.classList.add('scene-ready');
  animate(0);
}

function getShot() {
  const scroll = scrollState.position * Math.max(1, $('#story').offsetHeight - innerHeight * .5);
  let index = 0;
  while (index < anchors.length - 2 && scroll >= anchors[index + 1]) index++;
  let progress = clamp((scroll - anchors[index]) / Math.max(1, anchors[index + 1] - anchors[index]), 0, 1);
  // Hold the current composition before moving to the next chapter.
  progress = smooth(clamp((progress - (index === 4 ? .13 : .32)) / (index === 4 ? .87 : .68), 0, 1));
  if (reduceMotion.matches) progress = progress > .5 ? 1 : 0;
  const nearest = Math.min(4, index + (progress > .65 ? 1 : 0));
  activateChapter(nearest);
  if (index === 4 && progress > .25) $('#scene-status').textContent = '06 / BACK TO THE WHOLE';
  else if (index === 4) $('#scene-status').textContent = '05 / EXPLODED VIEW';
  return { index, progress, a: SHOTS[index], b: SHOTS[index + 1] };
}

function applyScene(time) {
  if (!ready) return;
  const { index, progress: t, a, b } = getShot();
  const mobile = mobileQuery.matches;
  const baseScale = lerp(a.scale, b.scale, t);
  // Moving the optical center keeps all parts in frame without fighting the copy.
  camera.clearViewOffset();
  camera.aspect = innerWidth / innerHeight;
  camera.fov = mobile ? 39 : 32;
  camera.position.set(7, 5.5, 11);
  target.set(...a.look).lerp(new THREE.Vector3(...b.look), t);
  camera.lookAt(target);
  camera.setViewOffset(innerWidth, innerHeight, -innerWidth * (mobile ? .0 : lerp(a.x, b.x, t)), -innerHeight * (mobile ? .26 : lerp(a.y, b.y, t)), innerWidth, innerHeight);
  camera.updateProjectionMatrix();
  const sceneScale = mobile ? baseScale * Math.min(innerHeight < 740 ? .57 : .68, .6 * (innerWidth / innerHeight) / .46) : baseScale;
  assembly.scale.setScalar(sceneScale);
  // Scale around the point of interest, keeping the stand and details centered.
  assembly.position.copy(target).multiplyScalar(1 - sceneScale);
  assembly.rotation.set(lerp(a.rx, b.rx, t), lerp(a.ry, b.ry, t) + (paused || reduceMotion.matches ? 0 : Math.sin(time * .3) * .045), 0);
  for (const id of Object.keys(PARTS)) {
    const from = a.parts[id], to = b.parts[id], model = meshes[id], material = materials[id];
    model.position.set(lerp(from.p[0], to.p[0], t), lerp(from.p[1], to.p[1], t), lerp(from.p[2], to.p[2], t));
    model.rotation.set(lerp(from.r[0], to.r[0], t), lerp(from.r[1], to.r[1], t), lerp(from.r[2], to.r[2], t));
    if (id === 'small_gear' && !paused && !reduceMotion.matches) model.rotation.y += Math.sin(time * .3) * .14;
    model.scale.setScalar(lerp(from.s, to.s, t));
    const opacity = lerp(from.a, to.a, t);
    model.visible = opacity > .015;
    if (material.transparent !== (opacity < .94)) { material.transparent = opacity < .94; material.needsUpdate = true; }
    material.opacity = opacity >= .94 ? 1 : opacity; material.depthWrite = !material.transparent;
    model.traverse((node) => { if (node.isMesh) node.castShadow = opacity > .8; });
  }
  const floorOpacity = index < 1 ? lerp(.7, .3, t) : index === 4 ? lerp(.2, .7, t) : .12;
  ring.material.opacity = floorOpacity * .3;
  shadow.material.opacity = floorOpacity * .5;
  scene.updateMatrixWorld(true);
  const labelShot = t > .65 ? Math.min(5, index + 1) : index;
  const displayedLabels = [[], ['frame', 'top_plate'], ['small_gear', 'bearing_holder', 'nut_holder'], ['camera_stand'], ['top_plate', 'frame', 'bearing_holder'], []][labelShot];
  for (const [id, label] of Object.entries(labels)) {
    const show = !mobile && displayedLabels.includes(id) && t < .2 || !mobile && displayedLabels.includes(id) && t > .85;
    label.element.style.opacity = show ? '1' : '0';
    if (!show) continue;
    meshes[id].getWorldPosition(world); world.project(camera);
    const px = (world.x * .5 + .5) * innerWidth;
    const py = (-world.y * .5 + .5) * innerHeight;
    let ox = id === 'nut_holder' ? 24 : id === 'camera_stand' ? 25 : 54;
    let oy = id === 'bearing_holder' ? 35 : id === 'small_gear' ? -35 : id === 'top_plate' ? -20 : 12;
    label.element.style.transform = `translate(${clamp(px + ox, 30, innerWidth - 210)}px, ${clamp(py + oy, 90, innerHeight - 90)}px)`;
  }
}

function animate(time) {
  if (failed) return;
  raf = requestAnimationFrame(animate);
  const elapsed = Math.min(.05, (time - previousTime) / 1000 || 0);
  previousTime = time;
  if (!visible || document.body.classList.contains('in-specifications')) return;
  const hasAmbientMotion = !paused && !reduceMotion.matches;
  if (hasAmbientMotion) ambientTime += elapsed;
  if (renderRequested || hasAmbientMotion) {
    applyScene(ambientTime);
    renderer.render(scene, camera);
    renderRequested = false;
  }
}

function resize() {
  if (renderer && !failed) {
    renderer.shadowMap.enabled = !mobileQuery.matches;
    renderer.setPixelRatio(Math.min(devicePixelRatio, mobileQuery.matches ? 1.5 : 2));
    renderer.setSize(innerWidth, innerHeight);
  }
  updateAnchors();
  ScrollTrigger.refresh();
}
window.addEventListener('resize', resize);
document.addEventListener('visibilitychange', () => { visible = !document.hidden; renderRequested = true; });
reduceMotion.addEventListener('change', () => {
  paused = reduceMotion.matches; updateMotionButton();
  scrollTween.scrollTrigger.scrubDuration(reduceMotion.matches ? 0 : .7);
  renderRequested = true;
});
document.fonts.ready.then(() => { updateAnchors(); ScrollTrigger.refresh(); });

try {
  setupScene();
  loadModels().catch((error) => { console.error('Unable to load the scanner models:', error); showFallback('STATIC PREVIEW'); });
} catch (error) {
  console.warn('3D graphics unavailable:', error);
  showFallback('STATIC PREVIEW');
}

window.addEventListener('pagehide', (event) => {
  if (event.persisted) return;
  cancelAnimationFrame(raf);
  ScrollTrigger.getAll().forEach((trigger) => trigger.kill());
  renderer?.dispose(); pmremTarget?.dispose();
});
window.addEventListener('pageshow', (event) => {
  if (event.persisted) { ScrollTrigger.refresh(); renderRequested = true; }
});
