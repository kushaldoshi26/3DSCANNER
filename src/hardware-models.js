import * as THREE from 'three';

/**
 * Procedural 3D model builder for all real hardware and electronic components
 * based on the physical 3D scanner assembly:
 * 1. 2020 Black Anodized Aluminum Extrusion Rail
 * 2. 608 Chrome Steel Ball Bearing (Outer ring, steel balls, inner ring)
 * 3. 28BYJ-48 Stepper Motor (Silver metal can, blue wire housing, brass shaft)
 * 4. Arduino Uno Microcontroller Board (Signature teal-blue PCB, USB-B port, chips, headers)
 * 5. ULN2003 Stepper Driver Module (Green PCB, LEDs, JST connector, IC)
 * 6. Motor Wiring Harness & USB Cable (Curved multi-color wiring)
 * 7. Fasteners: M3 Slotted pan-head screws and M4 hex nuts
 * 8. Optical Depth Camera mounted on top of the camera stand
 */

// Shared PBR Materials
const matBlackAnodized = new THREE.MeshStandardMaterial({
  color: 0x1c1e22,
  roughness: 0.38,
  metalness: 0.85,
});

const matChromeSteel = new THREE.MeshStandardMaterial({
  color: 0xf0f4f8,
  roughness: 0.14,
  metalness: 0.98,
  envMapIntensity: 1.6,
});

const matBrass = new THREE.MeshStandardMaterial({
  color: 0xd8b240,
  roughness: 0.25,
  metalness: 0.88,
});

const matSilverMetal = new THREE.MeshStandardMaterial({
  color: 0xd8dee4,
  roughness: 0.28,
  metalness: 0.86,
});

const matArduinoBlue = new THREE.MeshStandardMaterial({
  color: 0x00838f,
  roughness: 0.38,
  metalness: 0.08,
});

const matDriverGreen = new THREE.MeshStandardMaterial({
  color: 0x248a3d,
  roughness: 0.35,
  metalness: 0.08,
});

const matChipBlack = new THREE.MeshStandardMaterial({
  color: 0x141517,
  roughness: 0.65,
  metalness: 0.1,
});

const matGoldPins = new THREE.MeshStandardMaterial({
  color: 0xe5b839,
  roughness: 0.25,
  metalness: 0.9,
});

const matMotorBlue = new THREE.MeshStandardMaterial({
  color: 0x1868ba,
  roughness: 0.42,
  metalness: 0.06,
});

const matUsbBlue = new THREE.MeshStandardMaterial({
  color: 0x1976d2,
  roughness: 0.48,
  metalness: 0.05,
});

const matRedLed = new THREE.MeshStandardMaterial({
  color: 0xff2222,
  emissive: 0xff1111,
  emissiveIntensity: 0.8,
  roughness: 0.2,
});

// Helper: Slotted pan-head screw
function createSlottedScrew() {
  const screw = new THREE.Group();
  const head = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.07, 0.04, 20), matChromeSteel);
  head.castShadow = true;
  screw.add(head);

  const slot = new THREE.Mesh(new THREE.BoxGeometry(0.14, 0.025, 0.025), new THREE.MeshStandardMaterial({ color: 0x444444 }));
  slot.position.y = 0.02;
  screw.add(slot);
  return screw;
}

// Helper: Hex nut
function createHexNut() {
  const nutGeom = new THREE.CylinderGeometry(0.09, 0.09, 0.065, 6);
  const nut = new THREE.Mesh(nutGeom, matChromeSteel);
  nut.castShadow = true;

  const hole = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.075, 16), matBlackAnodized);
  nut.add(hole);
  return nut;
}

/**
 * 608 Ball Bearing + 2 mounting screws for Bearing Holder
 */
export function createBearingAssembly() {
  const group = new THREE.Group();
  group.name = 'bearing_assembly';

  // 608 Ball Bearing
  const bearing = new THREE.Group();

  // Outer Race (22mm outer)
  const outerRing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.245, 0.245, 0.16, 32, 1, true),
    matChromeSteel
  );
  bearing.add(outerRing);

  // Outer rim flat faces
  const outerTopRing = new THREE.Mesh(new THREE.RingGeometry(0.19, 0.245, 32), matChromeSteel);
  outerTopRing.rotation.x = -Math.PI / 2;
  outerTopRing.position.y = 0.08;
  bearing.add(outerTopRing);

  const outerBottomRing = outerTopRing.clone();
  outerBottomRing.position.y = -0.08;
  outerBottomRing.rotation.x = Math.PI / 2;
  bearing.add(outerBottomRing);

  // Inner Race (8mm bore)
  const innerRing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.09, 0.09, 0.16, 32, 1, true),
    matChromeSteel
  );
  bearing.add(innerRing);

  const innerTopRing = new THREE.Mesh(new THREE.RingGeometry(0.09, 0.13, 32), matChromeSteel);
  innerTopRing.rotation.x = -Math.PI / 2;
  innerTopRing.position.y = 0.08;
  bearing.add(innerTopRing);

  const innerBottomRing = innerTopRing.clone();
  innerBottomRing.position.y = -0.08;
  innerBottomRing.rotation.x = Math.PI / 2;
  bearing.add(innerBottomRing);

  // Metal shield ring
  const shieldMat = new THREE.MeshStandardMaterial({ color: 0x98a2aa, metalness: 0.85, roughness: 0.3 });
  const shield = new THREE.Mesh(new THREE.RingGeometry(0.132, 0.188, 32), shieldMat);
  shield.rotation.x = -Math.PI / 2;
  shield.position.y = 0.07;
  bearing.add(shield);

  // 8 Chrome Steel Ball Bearings
  const ballMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.98, roughness: 0.08 });
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.03, 12, 12), ballMat);
    ball.position.set(Math.cos(angle) * 0.16, 0, Math.sin(angle) * 0.16);
    bearing.add(ball);
  }

  // Position bearing concentric with bearing holder
  bearing.rotation.x = Math.PI / 2;
  bearing.position.set(0.08, 0.12, 0);
  group.add(bearing);

  // Two M3 mounting screws
  const screwLeft = createSlottedScrew();
  screwLeft.rotation.x = Math.PI / 2;
  screwLeft.position.set(-0.48, 0.14, 0);
  group.add(screwLeft);

  const screwRight = createSlottedScrew();
  screwRight.rotation.x = Math.PI / 2;
  screwRight.position.set(0.64, 0.14, 0);
  group.add(screwRight);

  return group;
}

/**
 * 2 Hex Nuts mounted inside the Nut Holder
 */
export function createNutHolderNuts() {
  const group = new THREE.Group();
  group.name = 'nut_holder_nuts';

  const nut1 = createHexNut();
  nut1.rotation.x = Math.PI / 2;
  nut1.position.set(-0.21, 0.05, 0);
  group.add(nut1);

  const nut2 = createHexNut();
  nut2.rotation.x = Math.PI / 2;
  nut2.position.set(0.21, 0.05, 0);
  group.add(nut2);

  return group;
}

/**
 * Stereo Camera Sensor mounted on top of Camera Stand
 */
export function createCameraSensor() {
  const group = new THREE.Group();
  group.name = 'camera_sensor';

  // Camera bar housing
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.72, 0.22, 0.24),
    new THREE.MeshStandardMaterial({ color: 0x202226, roughness: 0.35, metalness: 0.75 })
  );
  body.castShadow = true;
  group.add(body);

  // Front Glass Visor
  const visor = new THREE.Mesh(
    new THREE.BoxGeometry(0.68, 0.18, 0.02),
    new THREE.MeshStandardMaterial({ color: 0x06080c, roughness: 0.08, metalness: 0.95 })
  );
  visor.position.z = -0.125;
  group.add(visor);

  // Dual stereo lenses with AR coating
  const lensMat = new THREE.MeshStandardMaterial({ color: 0x081836, roughness: 0.12, metalness: 0.95 });
  const lensLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.02, 16), lensMat);
  lensLeft.rotation.x = Math.PI / 2;
  lensLeft.position.set(-0.24, 0, -0.135);
  group.add(lensLeft);

  const lensRight = lensLeft.clone();
  lensRight.position.x = 0.24;
  group.add(lensRight);

  // Center RGB sensor / IR dot projector
  const lensCenter = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.02, 16), matBrass);
  lensCenter.rotation.x = Math.PI / 2;
  lensCenter.position.set(0, 0, -0.135);
  group.add(lensCenter);

  // Mount bolt connecting camera to stand platform
  const mountScrew = createSlottedScrew();
  mountScrew.position.set(0, -0.12, 0);
  mountScrew.rotation.x = Math.PI;
  group.add(mountScrew);

  // Mount on the top platform of camera stand pointing down towards scanner
  group.position.set(0.08, 1.48, -0.18);
  group.rotation.set(-0.35, -0.42, 0.08);

  return group;
}

/**
 * Base internal electronics & structural rail:
 * - 2020 Aluminum Extrusion Rail
 * - 28BYJ-48 Stepper Motor
 * - Arduino Uno Board
 * - ULN2003 Driver Board
 * - Motor Wiring & USB Cable
 */
export function createBaseHardware() {
  const group = new THREE.Group();
  group.name = 'base_hardware';

  // ------------------------------------------
  // 1. 2020 BLACK ALUMINUM EXTRUSION RAIL
  // ------------------------------------------
  const railGroup = new THREE.Group();
  const railLength = 3.8;
  const railSize = 0.44; // 20mm scale

  const railMesh = new THREE.Mesh(
    new THREE.BoxGeometry(railLength, railSize, railSize),
    matBlackAnodized
  );
  railMesh.castShadow = true;
  railMesh.receiveShadow = true;
  railGroup.add(railMesh);

  // T-Slots
  const slotMat = new THREE.MeshStandardMaterial({ color: 0x0b0c0e, roughness: 0.8 });
  const topSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.05, 0.14), slotMat);
  topSlot.position.y = railSize / 2;
  railGroup.add(topSlot);

  const bottomSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.05, 0.14), slotMat);
  bottomSlot.position.y = -railSize / 2;
  railGroup.add(bottomSlot);

  const frontSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.14, 0.05), slotMat);
  frontSlot.position.z = railSize / 2;
  railGroup.add(frontSlot);

  const backSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.14, 0.05), slotMat);
  backSlot.position.z = -railSize / 2;
  railGroup.add(backSlot);

  // Silver core ends
  const coreMat = new THREE.MeshStandardMaterial({ color: 0xb5bcc4, metalness: 0.85, roughness: 0.4 });
  const endLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.02, 16), coreMat);
  endLeft.rotation.z = Math.PI / 2;
  endLeft.position.x = -railLength / 2;
  railGroup.add(endLeft);

  const endRight = new THREE.Mesh(new THREE.CylinderGeometry(0.08, 0.08, 0.02, 16), coreMat);
  endRight.rotation.z = Math.PI / 2;
  endRight.position.x = railLength / 2;
  railGroup.add(endRight);

  // Position rail through the frame to the camera stand
  railGroup.position.set(0.55, -1.33, 0);
  group.add(railGroup);

  // ------------------------------------------
  // 2. 28BYJ-48 STEPPER MOTOR
  // ------------------------------------------
  const motorGroup = new THREE.Group();
  motorGroup.name = 'stepper_motor';

  const motorCan = new THREE.Mesh(
    new THREE.CylinderGeometry(0.32, 0.32, 0.44, 32),
    matSilverMetal
  );
  motorCan.castShadow = true;
  motorGroup.add(motorCan);

  const motorPlate = new THREE.Mesh(
    new THREE.BoxGeometry(0.92, 0.035, 0.35),
    matSilverMetal
  );
  motorPlate.position.y = 0.22;
  motorPlate.castShadow = true;
  motorGroup.add(motorPlate);

  const screwLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.045, 0.05, 12), matChromeSteel);
  screwLeft.position.set(-0.38, 0.24, 0);
  motorGroup.add(screwLeft);

  const screwRight = screwLeft.clone();
  screwRight.position.set(0.38, 0.24, 0);
  motorGroup.add(screwRight);

  // Brass D-shaft
  const motorShaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.055, 0.055, 0.30, 16),
    matBrass
  );
  motorShaft.position.set(0, 0.36, 0);
  motorShaft.castShadow = true;
  motorGroup.add(motorShaft);

  // Blue wire box
  const motorWireBox = new THREE.Mesh(
    new THREE.BoxGeometry(0.34, 0.2, 0.26),
    matMotorBlue
  );
  motorWireBox.position.set(0, -0.15, -0.22);
  motorWireBox.castShadow = true;
  motorGroup.add(motorWireBox);

  // Directly under small gear [0.7, -1.3, -0.5]
  motorGroup.position.set(0.7, -1.62, -0.5);
  group.add(motorGroup);

  // ------------------------------------------
  // 3. ARDUINO UNO MICROCONTROLLER BOARD
  // ------------------------------------------
  const arduinoGroup = new THREE.Group();
  arduinoGroup.name = 'arduino_uno';

  const pcbW = 1.35, pcbH = 0.04, pcbD = 0.95;
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(pcbW, pcbH, pcbD), matArduinoBlue);
  pcb.castShadow = true;
  pcb.receiveShadow = true;
  arduinoGroup.add(pcb);

  // USB-B Port
  const usbPort = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.24, 0.26), matChromeSteel);
  usbPort.position.set(-pcbW / 2 + 0.14, 0.14, 0.25);
  usbPort.castShadow = true;
  arduinoGroup.add(usbPort);

  // DC Barrel Jack
  const dcJack = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.22, 0.22), matChipBlack);
  dcJack.position.set(-pcbW / 2 + 0.16, 0.13, -0.28);
  dcJack.castShadow = true;
  arduinoGroup.add(dcJack);

  // ATmega328P Chip
  const atmegaChip = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.07, 0.18), matChipBlack);
  atmegaChip.position.set(0.08, 0.05, 0.02);
  atmegaChip.castShadow = true;
  arduinoGroup.add(atmegaChip);

  // Headers
  const digitalHeader = new THREE.Mesh(new THREE.BoxGeometry(1.0, 0.16, 0.06), matChipBlack);
  digitalHeader.position.set(0.04, 0.1, 0.42);
  arduinoGroup.add(digitalHeader);

  const analogHeader = new THREE.Mesh(new THREE.BoxGeometry(0.88, 0.16, 0.06), matChipBlack);
  analogHeader.position.set(0.06, 0.1, -0.42);
  arduinoGroup.add(analogHeader);

  // Crystal
  const crystal = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.08, 0.09), matSilverMetal);
  crystal.position.set(-0.16, 0.05, 0.05);
  arduinoGroup.add(crystal);

  // Reset Button
  const resetBtn = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.07, 0.08), matRedLed);
  resetBtn.position.set(-pcbW / 2 + 0.12, 0.06, 0.42);
  arduinoGroup.add(resetBtn);

  // Position inside frame cavity
  arduinoGroup.position.set(-0.25, -1.68, 0.42);
  arduinoGroup.rotation.y = -0.15;
  group.add(arduinoGroup);

  // ------------------------------------------
  // 4. ULN2003 STEPPER DRIVER MODULE
  // ------------------------------------------
  const driverGroup = new THREE.Group();
  driverGroup.name = 'uln2003_driver';

  const driverPcb = new THREE.Mesh(new THREE.BoxGeometry(0.72, 0.035, 0.65), matDriverGreen);
  driverPcb.castShadow = true;
  driverPcb.receiveShadow = true;
  driverGroup.add(driverPcb);

  const ulnChip = new THREE.Mesh(new THREE.BoxGeometry(0.32, 0.05, 0.18), matChipBlack);
  ulnChip.position.set(0.02, 0.04, -0.06);
  ulnChip.castShadow = true;
  driverGroup.add(ulnChip);

  // 4 Red status LEDs
  for (let i = 0; i < 4; i++) {
    const led = new THREE.Mesh(new THREE.BoxGeometry(0.045, 0.045, 0.045), matRedLed);
    led.position.set(-0.24 + i * 0.09, 0.04, 0.2);
    driverGroup.add(led);
  }

  // White JST-XH motor connector
  const jst = new THREE.Mesh(
    new THREE.BoxGeometry(0.28, 0.14, 0.12),
    new THREE.MeshStandardMaterial({ color: 0xf5f5f5, roughness: 0.35 })
  );
  jst.position.set(0.18, 0.08, 0.18);
  driverGroup.add(jst);

  // Input pins
  const inPins = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.15, 0.05), matGoldPins);
  inPins.position.set(-0.16, 0.08, -0.22);
  driverGroup.add(inPins);

  driverGroup.position.set(1.05, -1.68, -0.08);
  driverGroup.rotation.y = 0.3;
  group.add(driverGroup);

  // ------------------------------------------
  // 5. MOTOR WIRING HARNESS & USB CABLE
  // ------------------------------------------
  const wireColors = [0x2196f3, 0xe91e63, 0xffeb3b, 0xff9800, 0xf44336];
  wireColors.forEach((color, idx) => {
    const offsetZ = (idx - 2) * 0.025;
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.7, -1.68, -0.68 + offsetZ),
      new THREE.Vector3(0.78, -1.58, -0.45 + offsetZ),
      new THREE.Vector3(0.95, -1.55, -0.25 + offsetZ),
      new THREE.Vector3(1.18, -1.62, 0.05 + offsetZ),
    ]);
    const wireGeom = new THREE.TubeGeometry(curve, 20, 0.016, 8, false);
    const wireMesh = new THREE.Mesh(wireGeom, new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
    wireMesh.castShadow = true;
    group.add(wireMesh);
  });

  // Blue USB Cable plugged into Arduino and trailing out through rear opening
  const usbCurve = new THREE.CatmullRomCurve3([
    new THREE.Vector3(-0.38, -1.54, 0.67),
    new THREE.Vector3(-0.48, -1.54, 0.85),
    new THREE.Vector3(-0.55, -1.52, 1.30),
    new THREE.Vector3(-0.62, -1.50, 1.75),
    new THREE.Vector3(-0.75, -1.50, 2.20),
  ]);
  const usbGeom = new THREE.TubeGeometry(usbCurve, 30, 0.045, 12, false);
  const usbMesh = new THREE.Mesh(usbGeom, matUsbBlue);
  usbMesh.castShadow = true;
  group.add(usbMesh);

  return {
    group,
    rail: railGroup,
    motor: motorGroup,
    arduino: arduinoGroup,
    driver: driverGroup,
  };
}
