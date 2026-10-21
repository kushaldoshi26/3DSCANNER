# 3D SCANNER

Interactive scanner website built with Vite, Three.js and GSAP.

## Latest website

- Text-only 3D SCANNER branding and shared Home, Workspace, About and account navigation.
- Eleven scanner components, including Arduino Uno, ULN2003 driver and 28BYJ-48 motor.
- Arduino USB alignment, inward-facing camera stand, rotating platter and linked gear animation.
- Sample vase on the platter and a replayable, simulated scan: 36-view capture, point-cloud reconstruction and final surface.
- Scroll-driven assembly, mechanism, camera stand, electronics and exploded views.
- Workspace with image upload, printable relief generation, GLB import, 3D preview, local project storage and STL/GLB export.

## Run locally

Use Node.js 22.12 or newer.

```sh
npm ci
npm run dev
```

For production, run `npm run build` and serve `dist/` over HTTP. All eight HTML entrypoints are built together. `render.yaml` includes the static hosting configuration.

## Camera and model generation

The sample scan illustrates the workflow; it does not connect to hardware or reconstruct camera images. ESP32-CAM with Arduino Uno is planned; phone-as-webcam support comes later. Actual capture controls remain unavailable.

The separate local companion optionally pairs with Arduino and connects the workspace to a separately installed TripoSR engine. Read [companion/README.md](companion/README.md) for setup. Engine weights are not included. The embossed relief tool works locally without the AI engine and is distinct from full object reconstruction.

Projects are stored in the browser's IndexedDB. STL exports use millimetres; GLB exports use metres. Physical fastener fit and actual hardware capture have not been validated.

## Authentication and hosting

The existing Sites publication uses platform authentication and owner-private access. Its Profile/Login navigation is specific to that host. Local previews and other static hosts show Login/Register; standalone authentication is not configured by this repository.

## Assets

Printed geometry and electronic board models come from the supplied source files. The motor, rail and bearing include visual reference geometry. The static fallback poster depicts an earlier assembly.
