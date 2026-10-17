import * as THREE from 'three';

/**
 * High-precision, realistic hardware & electronic components
 * engineered to fit accurately into the 3D scanner frame and assembly:
 * 1. 2020 Black Anodized Extrusion Rail (sits in the frame floor slot)
 * 2. 608 Ball Bearing (sits in the bearing holder socket)
 * 3. 28BYJ-48 Stepper Motor (sits in the molded motor cavity under small_gear)
 * 4. Arduino Uno R3 Board (sits flat on the frame floor with USB facing rear cutout)
 * 5. ULN2003 Stepper Driver Module (sits flat on the frame floor next to motor)
 * 6. Stereo Depth Camera (mounted securely to top of camera stand)
 * 7. Fasteners: M3 Pan-head screws and M4 Hex Nuts
 */

// PBR Materials
const matBlackAnodized = new THREE.MeshStandardMaterial({
  color: 0x1d1f23,
  roughness: 0.40,
  metalness: 0.82,
});

const matChromeSteel = new THREE.MeshStandardMaterial({
  color: 0xf2f6fa,
  roughness: 0.12,
  metalness: 0.98,
  envMapIntensity: 1.8,
});

const matBrass = new THREE.MeshStandardMaterial({
  color: 0xd6b038,
  roughness: 0.25,
  metalness: 0.88,
});

const matSilverMetal = new THREE.MeshStandardMaterial({
  color: 0xdce2e8,
  roughness: 0.26,
  metalness: 0.88,
});

const matArduinoBlue = new THREE.MeshStandardMaterial({
  color: 0x00838f,
  roughness: 0.35,
  metalness: 0.08,
});

const matDriverGreen = new THREE.MeshStandardMaterial({
  color: 0x22883c,
  roughness: 0.32,
  metalness: 0.08,
});

const matChipBlack = new THREE.MeshStandardMaterial({
  color: 0x151618,
  roughness: 0.65,
  metalness: 0.1,
});

const matGoldPins = new THREE.MeshStandardMaterial({
  color: 0xe5b839,
  roughness: 0.25,
  metalness: 0.9,
});

const matMotorBlue = new THREE.MeshStandardMaterial({
  color: 0x1a68ba,
  roughness: 0.42,
  metalness: 0.06,
});

const matUsbBlue = new THREE.MeshStandardMaterial({
  color: 0x1976d2,
  roughness: 0.45,
  metalness: 0.05,
});

const matRedLed = new THREE.MeshStandardMaterial({
  color: 0xff1744,
  emissive: 0xff1744,
  emissiveIntensity: 0.9,
  roughness: 0.2,
});

// Helper: Slotted pan-head screw
function createSlottedScrew(scale = 1) {
  const screw = new THREE.Group();
  const head = new THREE.Mesh(new THREE.CylinderGeometry(0.065 * scale, 0.055 * scale, 0.035 * scale, 18), matChromeSteel);
  head.castShadow = true;
  screw.add(head);

  const slot = new THREE.Mesh(new THREE.BoxGeometry(0.11 * scale, 0.02 * scale, 0.02 * scale), new THREE.MeshStandardMaterial({ color: 0x333333 }));
  slot.position.y = 0.018 * scale;
  screw.add(slot);
  return screw;
}

// Helper: Hex nut
function createHexNut(scale = 1) {
  const nutGeom = new THREE.CylinderGeometry(0.08 * scale, 0.08 * scale, 0.055 * scale, 6);
  const nut = new THREE.Mesh(nutGeom, matChromeSteel);
  nut.castShadow = true;

  const hole = new THREE.Mesh(new THREE.CylinderGeometry(0.04 * scale, 0.04 * scale, 0.065 * scale, 16), matBlackAnodized);
  nut.add(hole);
  return nut;
}

/**
 * 608 Ball Bearing + Mounting Screws for Bearing Holder
 */
export function createBearingAssembly() {
  const group = new THREE.Group();
  group.name = 'bearing_assembly';

  const bearing = new THREE.Group();

  // Outer Race (22mm outer)
  const outerRing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.24, 0.24, 0.15, 32, 1, true),
    matChromeSteel
  );
  bearing.add(outerRing);

  const outerTopRing = new THREE.Mesh(new THREE.RingGeometry(0.185, 0.24, 32), matChromeSteel);
  outerTopRing.rotation.x = -Math.PI / 2;
  outerTopRing.position.y = 0.075;
  bearing.add(outerTopRing);

  const outerBottomRing = outerTopRing.clone();
  outerBottomRing.position.y = -0.075;
  outerBottomRing.rotation.x = Math.PI / 2;
  bearing.add(outerBottomRing);

  // Inner Race (8mm bore)
  const innerRing = new THREE.Mesh(
    new THREE.CylinderGeometry(0.088, 0.088, 0.15, 32, 1, true),
    matChromeSteel
  );
  bearing.add(innerRing);

  const innerTopRing = new THREE.Mesh(new THREE.RingGeometry(0.088, 0.13, 32), matChromeSteel);
  innerTopRing.rotation.x = -Math.PI / 2;
  innerTopRing.position.y = 0.075;
  bearing.add(innerTopRing);

  const innerBottomRing = innerTopRing.clone();
  innerBottomRing.position.y = -0.075;
  innerBottomRing.rotation.x = Math.PI / 2;
  bearing.add(innerBottomRing);

  // Metal shield ring
  const shieldMat = new THREE.MeshStandardMaterial({ color: 0x9099a2, metalness: 0.88, roughness: 0.28 });
  const shield = new THREE.Mesh(new THREE.RingGeometry(0.132, 0.184, 32), shieldMat);
  shield.rotation.x = -Math.PI / 2;
  shield.position.y = 0.065;
  bearing.add(shield);

  // 8 Chrome Steel Ball Bearings
  const ballMat = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.98, roughness: 0.06 });
  for (let i = 0; i < 8; i++) {
    const angle = (i / 8) * Math.PI * 2;
    const ball = new THREE.Mesh(new THREE.SphereGeometry(0.028, 12, 12), ballMat);
    ball.position.set(Math.cos(angle) * 0.158, 0, Math.sin(angle) * 0.158);
    bearing.add(ball);
  }

  // Centered with bearing holder ring
  bearing.rotation.x = Math.PI / 2;
  bearing.position.set(0.08, 0.11, 0);
  group.add(bearing);

  // Two M3 mounting screws on bearing holder ears
  const screwLeft = createSlottedScrew(1);
  screwLeft.rotation.x = Math.PI / 2;
  screwLeft.position.set(-0.48, 0.13, 0);
  group.add(screwLeft);

  const screwRight = createSlottedScrew(1);
  screwRight.rotation.x = Math.PI / 2;
  screwRight.position.set(0.64, 0.13, 0);
  group.add(screwRight);

  return group;
}

/**
 * 2 Hex Nuts mounted inside the Nut Holder slots
 */
export function createNutHolderNuts() {
  const group = new THREE.Group();
  group.name = 'nut_holder_nuts';

  const nut1 = createHexNut(1);
  nut1.rotation.x = Math.PI / 2;
  nut1.position.set(-0.21, 0.045, 0);
  group.add(nut1);

  const nut2 = createHexNut(1);
  nut2.rotation.x = Math.PI / 2;
  nut2.position.set(0.21, 0.045, 0);
  group.add(nut2);

  return group;
}

/**
 * Stereo Depth Camera Sensor mounted on top of Camera Stand
 */
export function createCameraSensor() {
  const group = new THREE.Group();
  group.name = 'camera_sensor';

  // Camera bar housing
  const body = new THREE.Mesh(
    new THREE.BoxGeometry(0.70, 0.20, 0.22),
    new THREE.MeshStandardMaterial({ color: 0x1f2125, roughness: 0.35, metalness: 0.75 })
  );
  body.castShadow = true;
  group.add(body);

  // Front Glass Visor
  const visor = new THREE.Mesh(
    new THREE.BoxGeometry(0.66, 0.16, 0.02),
    new THREE.MeshStandardMaterial({ color: 0x05070a, roughness: 0.08, metalness: 0.95 })
  );
  visor.position.z = -0.115;
  group.add(visor);

  // Dual stereo lenses with AR coating
  const lensMat = new THREE.MeshStandardMaterial({ color: 0x091834, roughness: 0.10, metalness: 0.95 });
  const lensLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.042, 0.042, 0.02, 16), lensMat);
  lensLeft.rotation.x = Math.PI / 2;
  lensLeft.position.set(-0.23, 0, -0.125);
  group.add(lensLeft);

  const lensRight = lensLeft.clone();
  lensRight.position.x = 0.23;
  group.add(lensRight);

  // Center RGB sensor / IR dot projector
  const lensCenter = new THREE.Mesh(new THREE.CylinderGeometry(0.032, 0.032, 0.02, 16), matBrass);
  lensCenter.rotation.x = Math.PI / 2;
  lensCenter.position.set(0, 0, -0.125);
  group.add(lensCenter);

  // Mounting screw at base of camera
  const mountScrew = createSlottedScrew(0.9);
  mountScrew.position.set(0, -0.11, 0);
  mountScrew.rotation.x = Math.PI;
  group.add(mountScrew);

  // Mounted on the camera stand top head pointing at scanner platter
  group.position.set(0.08, 1.48, -0.18);
  group.rotation.set(-0.35, -0.42, 0.08);

  return group;
}

/**
 * Clean internal hardware components positioned accurately inside the frame:
 * - 2020 Aluminum Rail (seated in frame floor channel)
 * - 28BYJ-48 Stepper Motor (seated in motor cavity below small_gear)
 * - Arduino Uno (flat on frame floor, rear USB cutout)
 * - ULN2003 Driver (flat on frame floor beside motor)
 */
export function createInternalHardware() {
  const group = new THREE.Group();
  group.name = 'internal_hardware';

  // ------------------------------------------
  // 1. 2020 BLACK ALUMINUM EXTRUSION RAIL
  // ------------------------------------------
  const railGroup = new THREE.Group();
  railGroup.name = 'aluminum_rail';
  const railLength = 3.6;
  const railSize = 0.40; // 20mm scaled to match frame channel

  const railMesh = new THREE.Mesh(
    new THREE.BoxGeometry(railLength, railSize, railSize),
    matBlackAnodized
  );
  railMesh.castShadow = true;
  railMesh.receiveShadow = true;
  railGroup.add(railMesh);

  // T-Slots on 4 faces
  const slotMat = new THREE.MeshStandardMaterial({ color: 0x0c0d0f, roughness: 0.75 });
  const topSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.04, 0.12), slotMat);
  topSlot.position.y = railSize / 2;
  railGroup.add(topSlot);

  const bottomSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.04, 0.12), slotMat);
  bottomSlot.position.y = -railSize / 2;
  railGroup.add(bottomSlot);

  const frontSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.12, 0.04), slotMat);
  frontSlot.position.z = railSize / 2;
  railGroup.add(frontSlot);

  const backSlot = new THREE.Mesh(new THREE.BoxGeometry(railLength, 0.12, 0.04), slotMat);
  backSlot.position.z = -railSize / 2;
  railGroup.add(backSlot);

  // Silver core ends
  const coreMat = new THREE.MeshStandardMaterial({ color: 0xb5bcc4, metalness: 0.85, roughness: 0.4 });
  const endLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.02, 16), coreMat);
  endLeft.rotation.z = Math.PI / 2;
  endLeft.position.x = -railLength / 2;
  railGroup.add(endLeft);

  const endRight = new THREE.Mesh(new THREE.CylinderGeometry(0.07, 0.07, 0.02, 16), coreMat);
  endRight.rotation.z = Math.PI / 2;
  endRight.position.x = railLength / 2;
  railGroup.add(endRight);

  // Position rail in the frame floor slot (y = -1.55), passing through to camera stand
  railGroup.position.set(0.60, -1.55, 0);
  group.add(railGroup);

  // ------------------------------------------
  // 2. 28BYJ-48 STEPPER MOTOR
  // ------------------------------------------
  const motorGroup = new THREE.Group();
  motorGroup.name = 'stepper_motor';

  const motorCan = new THREE.Mesh(
    new THREE.CylinderGeometry(0.28, 0.28, 0.38, 32),
    matSilverMetal
  );
  motorCan.castShadow = true;
  motorGroup.add(motorCan);

  // Mounting ears
  const motorPlate = new THREE.Mesh(
    new THREE.BoxGeometry(0.84, 0.03, 0.30),
    matSilverMetal
  );
  motorPlate.position.y = 0.19;
  motorPlate.castShadow = true;
  motorGroup.add(motorPlate);

  const screwLeft = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, 0.04, 12), matChromeSteel);
  screwLeft.position.set(-0.34, 0.21, 0);
  motorGroup.add(screwLeft);

  const screwRight = screwLeft.clone();
  screwRight.position.set(0.34, 0.21, 0);
  motorGroup.add(screwRight);

  // Brass D-shaft extending up into small gear
  const motorShaft = new THREE.Mesh(
    new THREE.CylinderGeometry(0.05, 0.05, 0.26, 16),
    matBrass
  );
  motorShaft.position.set(0, 0.32, 0);
  motorShaft.castShadow = true;
  motorGroup.add(motorShaft);

  // Blue wire cap at rear
  const motorWireBox = new THREE.Mesh(
    new THREE.BoxGeometry(0.30, 0.18, 0.22),
    matMotorBlue
  );
  motorWireBox.position.set(0, -0.12, -0.19);
  motorWireBox.castShadow = true;
  motorGroup.add(motorWireBox);

  // Position motor seated inside the frame pocket right under small gear [0.7, -1.3, -0.5]
  motorGroup.position.set(0.7, -1.62, -0.5);
  group.add(motorGroup);

  // ------------------------------------------
  // 3. ARDUINO UNO MICROCONTROLLER BOARD
  // ------------------------------------------
  const arduinoGroup = new THREE.Group();
  arduinoGroup.name = 'arduino_uno';

  const pcbW = 1.25, pcbH = 0.035, pcbD = 0.88;
  const pcb = new THREE.Mesh(new THREE.BoxGeometry(pcbW, pcbH, pcbD), matArduinoBlue);
  pcb.castShadow = true;
  pcb.receiveShadow = true;
  arduinoGroup.add(pcb);

  // Silver USB-B Port
  const usbPort = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.20, 0.22), matChromeSteel);
  usbPort.position.set(-pcbW / 2 + 0.12, 0.12, 0.22);
  usbPort.castShadow = true;
  arduinoGroup.add(usbPort);

  // Black DC Barrel Jack
  const dcJack = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.18, 0.18), matChipBlack);
  dcJack.position.set(-pcbW / 2 + 0.14, 0.11, -0.24);
  dcJack.castShadow = true;
  arduinoGroup.add(dcJack);

  // ATmega328P Chip
  const atmegaChip = new THREE.Mesh(new THREE.BoxGeometry(0.60, 0.06, 0.16), matChipBlack);
  atmegaChip.position.set(0.06, 0.045, 0.02);
  atmegaChip.castShadow = true;
  arduinoGroup.add(atmegaChip);

  // Pin Headers
  const digitalHeader = new THREE.Mesh(new THREE.BoxGeometry(0.85, 0.14, 0.05), matChipBlack);
  digitalHeader.position.set(0.06, 0.09, 0.38);
  arduinoGroup.add(digitalHeader);

  const analogHeader = new THREE.Mesh(new THREE.BoxGeometry(0.75, 0.14, 0.05), matChipBlack);
  analogHeader.position.set(0.08, 0.09, -0.38);
  arduinoGroup.add(analogHeader);

  // Crystal
  const crystal = new THREE.Mesh(new THREE.BoxGeometry(0.15, 0.07, 0.08), matSilverMetal);
  crystal.position.set(-0.14, 0.045, 0.05);
  arduinoGroup.add(crystal);

  // Reset Button
  const resetBtn = new THREE.Mesh(new THREE.BoxGeometry(0.07, 0.06, 0.07), matRedLed);
  resetBtn.position.set(-pcbW / 2 + 0.10, 0.05, 0.38);
  arduinoGroup.add(resetBtn);

  // Seated flat on the floor of the frame
  arduinoGroup.position.set(-0.35, -1.72, 0.38);
  arduinoGroup.rotation.y = -0.15;
  group.add(arduinoGroup);

  // ------------------------------------------
  // 4. ULN2003 STEPPER DRIVER MODULE
  // ------------------------------------------
  const driverGroup = new THREE.Group();
  driverGroup.name = 'uln2003_driver';

  const driverPcb = new THREE.Mesh(new THREE.BoxGeometry(0.65, 0.03, 0.58), matDriverGreen);
  driverPcb.castShadow = true;
  driverPcb.receiveShadow = true;
  driverGroup.add(driverPcb);

  const ulnChip = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.045, 0.16), matChipBlack);
  ulnChip.position.set(0.02, 0.035, -0.05);
  ulnChip.castShadow = true;
  driverGroup.add(ulnChip);

  // 4 Red status LEDs
  for (let i = 0; i < 4; i++) {
    const led = new THREE.Mesh(new THREE.BoxGeometry(0.04, 0.04, 0.04), matRedLed);
    led.position.set(-0.21 + i * 0.08, 0.035, 0.18);
    driverGroup.add(led);
  }

  // White JST-XH connector
  const jst = new THREE.Mesh(
    new THREE.BoxGeometry(0.24, 0.12, 0.10),
    new THREE.MeshStandardMaterial({ color: 0xf0f0f0, roughness: 0.35 })
  );
  jst.position.set(0.16, 0.07, 0.16);
  driverGroup.add(jst);

  // Seated flat on frame floor next to motor
  driverGroup.position.set(0.92, -1.72, 0.05);
  driverGroup.rotation.y = 0.25;
  group.add(driverGroup);

  // ------------------------------------------
  // 5. CLEAN MOTOR WIRING
  // ------------------------------------------
  const wireColors = [0x2196f3, 0xe91e63, 0xffeb3b, 0xff9800, 0xf44336];
  wireColors.forEach((color, idx) => {
    const offsetZ = (idx - 2) * 0.02;
    const curve = new THREE.CatmullRomCurve3([
      new THREE.Vector3(0.70, -1.68, -0.65 + offsetZ),
      new THREE.Vector3(0.76, -1.68, -0.45 + offsetZ),
      new THREE.Vector3(0.85, -1.69, -0.22 + offsetZ),
      new THREE.Vector3(0.98, -1.69, 0.08 + offsetZ),
    ]);
    const wireGeom = new THREE.TubeGeometry(curve, 16, 0.014, 8, false);
    const wireMesh = new THREE.Mesh(wireGeom, new THREE.MeshStandardMaterial({ color, roughness: 0.4 }));
    wireMesh.castShadow = true;
    group.add(wireMesh);
  });

  return {
    group,
    rail: railGroup,
    motor: motorGroup,
    arduino: arduinoGroup,
    driver: driverGroup,
  };
}
