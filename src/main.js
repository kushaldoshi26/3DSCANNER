import {createScanDemo} from './scan-demo.js';
import {navigation} from './navigation.js';
document.querySelector('#global-navigation').innerHTML=navigation('home');
import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { gsap } from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

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

const PARTS = {
  aluminium_rail: {label:'2020 CAMERA RAIL',number:'10',native:true,procedural:true,position:[3.10,-1.43,0],rotation:[0,0,0]},
  ball_bearing: {label:'608 BEARING',number:'11',native:true,procedural:true,position:[0,-1.003,0],rotation:[0,0,0]},
  frame: { label: 'MAIN FRAME', number: '01', color: '#56666b', roughness: .35, metalness: .32, position: [0, -1.4, 0], rotation: [halfPi, 0, 0] },
  top_plate: { label: 'TOP PLATE', number: '02', color: '#d3c3b9', roughness: .3, metalness: .35, position: [0, -.93, 0], rotation: [0, 0, 0] },
  small_gear: { label: 'SMALL GEAR', number: '03', color: '#b58863', roughness: .29, metalness: .5, position: [.184, -.99, -.966], rotation: [0, 0, 0] },
  nut_holder: { label: 'NUT HOLDER', number: '04', color: '#d3c3b9', roughness: .34, metalness: .3, position: [6.472, -1.60, 0], rotation: [halfPi, 0, 0] },
  bearing_holder: { label: 'BEARING HOLDER', number: '05', color: '#b58863', roughness: .32, metalness: .4, position: [.08676, -1.04925, 0], rotation: [halfPi, 0, 0] },
  arduino_uno: { label: 'ARDUINO UNO R3', number: '07', native: true, position: [.08717, -1.50462, .718], rotation: [0, 0, 0] },
  stepper_motor: { label: '28BYJ-48 MOTOR', number: '08', native: true, position: [0, -1.245, -.9315], rotation: [0, 0, 0] },
  stepper_driver: { label: 'ULN2003 DRIVER', number: '09', native: true, position: [-.96, -1.44, .25], rotation: [0, 0, 0] },
  camera_stand: { label: 'CAMERA STAND', number: '06', color: '#64747a', roughness: .38, metalness: .35, position: [6.4, .85206, 0], rotation: [Math.PI - Math.PI / 8, halfPi, 0] },
};

// USB face centre matches the frame cutout at (-7.56, -2.25) mm.
// Stand is inverted and its 22.5-degree mounting plane levelled onto the rail.
// Physical screw fit remains unverified.
// Coordinates use the original STL numerical units at a common 0.023 scale.
function partPose(name, overrides = {}) {
  return { p: PARTS[name].position, r: PARTS[name].rotation, s: 1, a: 1, ...overrides };
}
const assembledParts = () => Object.fromEntries(Object.keys(PARTS).map((id) => [id, partPose(id)]));
// Reference assembly: the motor pocket is at (0, -42 mm) in plan;
// the shaft is 8 mm eccentric. The bearing holder is offset 3.772 mm
// so its bearing bore sits on the frame origin. The nut holder belongs to the rail.
const SHOTS = [
 {x:.22,y:0,scale:.52,ry:.65,rx:-.06,look:[2.65,.6,0],parts:assembledParts()},
 {x:.23,y:0,scale:.92,ry:.3,rx:.16,look:[.7,.2,0],parts:{...assembledParts(),
   top_plate:partPose('top_plate',{p:[0,1.3,0],r:[-.8,0,0]}),
   camera_stand:partPose('camera_stand',{a:.04}),aluminium_rail:partPose('aluminium_rail',{a:0}),nut_holder:partPose('nut_holder',{a:.04}),
 }},
 {x:-.22,y:0,scale:1.18,ry:-.1,rx:.2,look:[0,-.2,0],parts:{...assembledParts(),
   frame:partPose('frame',{a:.06}),top_plate:partPose('top_plate',{p:[0,2,0],a:.05}),
   small_gear:partPose('small_gear',{p:[.184,.2,-.966]}),
   bearing_holder:partPose('bearing_holder',{p:[.08676,.5,0]}),ball_bearing:partPose('ball_bearing',{p:[0,.62,0]}),
   camera_stand:partPose('camera_stand',{a:0}),aluminium_rail:partPose('aluminium_rail',{a:.04}),nut_holder:partPose('nut_holder',{a:0}),
   arduino_uno:partPose('arduino_uno',{a:.1}),stepper_driver:partPose('stepper_driver',{a:.1}),
 }},
 {x:.23,y:0,scale:.95,ry:1.1,rx:0,look:[6.1,.5,0],parts:{...assembledParts(),
   frame:partPose('frame',{a:0}),top_plate:partPose('top_plate',{a:0}),small_gear:partPose('small_gear',{a:0}),bearing_holder:partPose('bearing_holder',{a:0}),ball_bearing:partPose('ball_bearing',{a:0}),
   arduino_uno:partPose('arduino_uno',{a:0}),stepper_driver:partPose('stepper_driver',{a:0}),stepper_motor:partPose('stepper_motor',{a:0}),
   nut_holder:partPose('nut_holder',{p:[6.472,-2.05,0]}),
 }},
 {x:.23,y:0,scale:1.18,ry:.2,rx:.35,look:[0,-.25,0],parts:{...assembledParts(),
   frame:partPose('frame',{a:.09}),top_plate:partPose('top_plate',{p:[0,2.1,0],a:0}),bearing_holder:partPose('bearing_holder',{p:[.08676,1.5,0],a:.07}),ball_bearing:partPose('ball_bearing',{a:.07}),
   camera_stand:partPose('camera_stand',{a:0}),aluminium_rail:partPose('aluminium_rail',{a:.05}),nut_holder:partPose('nut_holder',{a:0}),
   small_gear:partPose('small_gear',{p:[.184,-.1,-.966]}),
   arduino_uno:partPose('arduino_uno',{p:[.08717,-.25,.718]}),stepper_driver:partPose('stepper_driver',{p:[-.96,-.25,.25]}),stepper_motor:partPose('stepper_motor',{p:[0,-.41,-.9315]}),
 }},
 {x:.22,y:0,scale:.52,ry:.65,rx:.12,look:[2.65,.6,0],parts:{...assembledParts(),
   top_plate:partPose('top_plate',{p:[0,2,0]}),bearing_holder:partPose('bearing_holder',{p:[.08676,.5,0]}),ball_bearing:partPose('ball_bearing',{p:[0,.65,0]}),
   small_gear:partPose('small_gear',{p:[.184,.15,-.966]}),stepper_motor:partPose('stepper_motor',{p:[0,-.41,-.9315]}),
   arduino_uno:partPose('arduino_uno',{p:[.08717,-.6,.718]}),stepper_driver:partPose('stepper_driver',{p:[-.96,-.55,.25]}),
   nut_holder:partPose('nut_holder',{p:[6.472,-2.1,0]}),
 }},
 {x:.22,y:0,scale:.52,ry:.65,rx:-.06,look:[2.65,.6,0],parts:assembledParts()},
];
const lastChapter = chapters.length - 1;

function updateMotionButton() {
  controls.disabled = reduceMotion.matches || failed;
  if (reduceMotion.matches) {
    controls.setAttribute('aria-pressed', 'true');
    controls.setAttribute('aria-label', 'Platter animation disabled by your reduced motion preference');
    controls.title = 'Reduced motion is enabled on your device';
    return;
  }
  controls.setAttribute('aria-pressed', String(paused));
  controls.setAttribute('aria-label', paused ? 'Resume platter animation' : 'Pause platter animation');
  controls.title = paused ? 'Resume platter animation' : 'Pause platter animation';
}
updateMotionButton();
controls.addEventListener('click', () => { paused = !paused; updateMotionButton(); renderRequested = true; });
$('#reset-view').addEventListener('click', () => {
  window.scrollTo({ top: 0, behavior: reduceMotion.matches ? 'instant' : 'smooth' });
  renderRequested = true;
});

function updateAnchors() {
  const vh = innerHeight;
  anchors = chapters.map((el, i) => i === 0 ? 0 : el.offsetTop);
  anchors.push(Math.max(anchors.at(-1) + vh, $('#specifications').offsetTop - vh * 1.08));
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
  const labels = ['01 / ASSEMBLY', '02 / STRUCTURE', '03 / MECHANISM', '04 / PERSPECTIVE', '05 / ELECTRONICS', '06 / EXPLODED VIEW'];
  $('#scene-status').textContent = labels[index];
}

// One scrubbed timeline is the source of truth in either scroll direction.
const scrollTween = gsap.fromTo(scrollState, { position: 0 }, {
  position: 1, duration: 1, ease: 'none',
  onUpdate: () => { renderRequested = true; },
  scrollTrigger: {
    trigger: '#story', start: 'top top', end: 'bottom center',
    scrub: reduceMotion.matches ? true : .45,
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
let scanDemo;
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
  renderer.shadowMap.type = THREE.PCFShadowMap;
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

function proceduralPart(id) {
 const g=new THREE.Group();const metal=new THREE.MeshStandardMaterial({color:'#a8b1b4',metalness:.8,roughness:.28});
 if(id==='ball_bearing') {
  const shape=new THREE.Shape();shape.absarc(0,0,11,0,Math.PI*2,false);const hole=new THREE.Path();hole.absarc(0,0,4,0,Math.PI*2,true);shape.holes.push(hole);
  const mesh=new THREE.Mesh(new THREE.ExtrudeGeometry(shape,{depth:7,bevelEnabled:false,curveSegments:48}),metal);mesh.rotation.x=-halfPi;g.add(mesh);
 } else {
  // Continuous 2020 T-slot cross-section with a central bore.
  const section=new THREE.Shape();
  const side=[[-10,10],[-3,10],[-3,8],[-6,8],[-6,6],[-3,3],[3,3],[6,6],[6,8],[3,8],[3,10],[10,10]];
  for(let turn=0;turn<4;turn++)for(let i=turn?1:0;i<side.length;i++){
   let [x,y]=side[i];for(let r=0;r<turn;r++)[x,y]=[y,-x];
   if(turn===0&&i===0)section.moveTo(x,y);else section.lineTo(x,y);
  }
  section.closePath();const bore=new THREE.Path();bore.absarc(0,0,2.1,0,Math.PI*2,true);section.holes.push(bore);
  const profile=new THREE.Mesh(new THREE.ExtrudeGeometry(section,{depth:300,steps:1,bevelEnabled:false,curveSegments:24}),metal);
  profile.rotation.y=halfPi;g.add(profile);
 }
 return g;
}

async function loadModels() {
  const loader = new GLTFLoader();
  let loaded = 0;
  await Promise.all(Object.entries(PARTS).map(async ([id, part]) => {
    const gltf = part.procedural ? {scene:proceduralPart(id)} : await loader.loadAsync(modelUrl(`${id}.glb`));
    const model = gltf.scene;
    model.updateMatrixWorld(true);
    const box = new THREE.Box3().setFromObject(model);
    const center = box.getCenter(new THREE.Vector3());
    model.position.sub(center);
    const partMaterials = [];
    const printedMaterial = part.native ? null : new THREE.MeshStandardMaterial({ color: part.color, roughness: part.roughness, metalness: part.metalness, envMapIntensity: 1.1 });
    model.traverse(node => {
      if (!node.isMesh) return;
      if (printedMaterial) { node.material.dispose?.(); node.material = printedMaterial; }
      else node.material = Array.isArray(node.material) ? node.material.map(m => m.clone()) : node.material.clone();
      for (const material of Array.isArray(node.material) ? node.material : [node.material]) {
        if (!partMaterials.includes(material)) partMaterials.push(material);
      }
      node.castShadow = true; node.receiveShadow = true;
    });
    const normalized = new THREE.Group(); normalized.add(model); normalized.scale.setScalar(.023);
    if (id === 'top_plate') {
      const indexMark = new THREE.Mesh(new THREE.BoxGeometry(8, .18, 1.2), new THREE.MeshStandardMaterial({color:'#b58863',roughness:.5}));
      indexMark.position.set(55, 6.12, 0); normalized.add(indexMark); partMaterials.push(indexMark.material);
    }
    const pivot = new THREE.Group(); pivot.add(normalized);
    assembly.add(pivot); meshes[id] = pivot; materials[id] = partMaterials;
    createLabel(id);
    loaded++;
    $('#load-progress').style.width = `${loaded / Object.keys(PARTS).length * 100}%`;
    $('#load-label').textContent = `${String(loaded).padStart(2, '0')} / ${String(Object.keys(PARTS).length).padStart(2, '0')} PARTS READY`;
  }));
  if (failed) return;
  scanDemo = createScanDemo(assembly, meshes.top_plate, () => {renderRequested=true;}, () => {if(!reduceMotion.matches){paused=false;updateMotionButton();}});
  $('#scan-demo-start').disabled=reduceMotion.matches;
  if(reduceMotion.matches) $('#scan-demo-start').textContent='Sample vase · motion reduced';
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
  progress = smooth(clamp((progress - (index === lastChapter ? .13 : .18)) / (index === lastChapter ? .87 : .82), 0, 1));
  if (reduceMotion.matches) progress = progress > .5 ? 1 : 0;
  const nearest = Math.min(lastChapter, index + (progress > .65 ? 1 : 0));
  activateChapter(nearest);
  if (index === lastChapter && progress > .25) $('#scene-status').textContent = '07 / BACK TO THE WHOLE';
  else if (index === lastChapter) $('#scene-status').textContent = '06 / EXPLODED VIEW';
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
  assembly.rotation.set(lerp(a.rx, b.rx, t), lerp(a.ry, b.ry, t) + (paused || reduceMotion.matches ? 0 : Math.sin(time * .3) * .045), 0);
  // Rotate and scale about the current point of interest, including the long rail.
  assembly.position.copy(target).sub(target.clone().applyEuler(assembly.rotation).multiplyScalar(sceneScale));
  for (const id of Object.keys(PARTS)) {
    const from = a.parts[id], to = b.parts[id], model = meshes[id], material = materials[id];
    model.position.set(lerp(from.p[0], to.p[0], t), lerp(from.p[1], to.p[1], t), lerp(from.p[2], to.p[2], t));
    model.rotation.set(lerp(from.r[0], to.r[0], t), lerp(from.r[1], to.r[1], t), lerp(from.r[2], to.r[2], t), 'YXZ');
    // Local platter spin follows its own axis even while the exploded shot tilts.
    if (id === 'top_plate' || id === 'small_gear') model.rotateY((scanDemo?.angle(time) ?? time * .28) * (id === 'small_gear' ? 5 : 1));
    model.scale.setScalar(lerp(from.s, to.s, t));
    const opacity = lerp(from.a, to.a, t);
    model.visible = opacity > .015;
    for (const mat of material) {
      if (mat.transparent !== (opacity < .94)) { mat.transparent = opacity < .94; mat.needsUpdate = true; }
      mat.opacity = opacity >= .94 ? 1 : opacity; mat.depthWrite = !mat.transparent;
    }
    model.traverse((node) => { if (node.isMesh) node.castShadow = opacity > .8; });
  }
  scanDemo?.update(time, index === 0 && t < .12 || index === lastChapter && t > .98);
  const floorOpacity = index < 1 ? lerp(.7, .3, t) : index === lastChapter ? lerp(.2, .7, t) : .12;
  ring.material.opacity = floorOpacity * .3;
  shadow.material.opacity = floorOpacity * .5;
  scene.updateMatrixWorld(true);
  const labelShot = t > .65 ? Math.min(SHOTS.length - 1, index + 1) : index;
  const displayedLabels = [[], ['frame', 'top_plate'], ['small_gear', 'bearing_holder', 'stepper_motor'], ['camera_stand', 'nut_holder'], ['arduino_uno', 'stepper_motor', 'stepper_driver'], ['top_plate', 'bearing_holder', 'stepper_motor'], []][labelShot];
  for (const [id, label] of Object.entries(labels)) {
    const show = !mobile && displayedLabels.includes(id) && t < .2 || !mobile && displayedLabels.includes(id) && t > .85;
    label.element.style.opacity = show ? '1' : '0';
    if (!show) continue;
    meshes[id].getWorldPosition(world); world.project(camera);
    const px = (world.x * .5 + .5) * innerWidth;
    const py = (-world.y * .5 + .5) * innerHeight;
    let ox = id === 'nut_holder' ? 24 : id === 'camera_stand' ? 25 : 54;
    let oy = id === 'stepper_driver' ? 65 : id === 'arduino_uno' ? -35 : id === 'bearing_holder' ? 35 : id === 'small_gear' ? -35 : id === 'top_plate' ? -20 : 12;
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
  scrollTween.scrollTrigger.scrubDuration(reduceMotion.matches ? 0 : .45);
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
