# FORM / 01 — 3D Scanner Interactive Showcase

[![GitHub Repo](https://img.shields.io/badge/GitHub-3DSCANNER-blue?logo=github)](https://github.com/kushaldoshi26/3DSCANNER)
[![Three.js](https://img.shields.io/badge/Three.js-r186-black?logo=three.js)](https://threejs.org/)
[![Vite](https://img.shields.io/badge/Vite-8.3-646CFF?logo=vite)](https://vitejs.dev/)
[![GSAP](https://img.shields.io/badge/GSAP-ScrollTrigger-green)](https://greensock.com/)

A premium, scroll-driven interactive 3D product showcase for an open-source 3D scanner. Built with **Three.js**, **GSAP ScrollTrigger**, and **Vite**, this application transforms static 3D printer STL geometry into a cinematic, educational assembly experience.

---

## 📸 Overview

The website deconstructs the physical 3D scanner into five interactive chapters: from the fully assembled functional device to an exploded view exposing internal drive mechanisms, bearings, fasteners, and sensor mounts.

---

## 🚀 Key Features & What Was Implemented

### 1. 🎨 Curated 3D PBR Color Palette & Materials
Every component is styled with physically-based rendering (PBR) materials, realistic roughness, metalness, and ambient reflections from an HDR studio environment (`RoomEnvironment` via `PMREMGenerator`):

| Component | Color / Material | Visual Description |
| :--- | :--- | :--- |
| **01. Main Frame** | `#445761` (Slate Blue) | The central circular printed foundation with wire & extrusion pass-through slots |
| **02. Top Plate** | `#ded5c3` (Warm Ivory) | The turn-table platter with an integrated gear ring underneath |
| **03. Small Gear** | `#d97b2b` (Copper Orange) | High-contrast precision drive gear engaging with the platter ring |
| **04. Nut Holder** | `#e5ded2` (Technical Porcelain) | Printed T-nut block with two pressed-in chrome hex nuts |
| **05. Bearing Holder**| `#b88344` (Amber Bronze) | Central structural bridge supporting the turntable rotation |
| **06. Camera Stand** | `#4e626c` (Slate Titanium) | Articulated upright arm positioning the depth sensor |
| **07. 608 Ball Bearing** | Chrome Steel (`metalness: 0.98`) | High-polish outer race, 8 spherical steel ball bearings, and inner ring |
| **08. Fasteners** | Polished Chrome Steel | M3 slotted pan-head mounting screws and M4 hex nuts |
| **09. Camera Sensor** | Matte Black & Glass | Optical stereo depth sensor bar mounted atop the camera arm |

---

### 2. 🎬 5-Stage Interactive Scroll Journey (GSAP ScrollTrigger)
The page uses scroll position as a single scrubbed timeline to navigate five distinct visual perspectives:

- **Chapter 01: The Complete Form** (`01 / ASSEMBLY`)
  - Studio view of the assembled scanner rotating slowly with ambient lighting and soft radial contact shadows.
- **Chapter 02: The Structure** (`02 / STRUCTURE`)
  - The top platter lifts upward and angles forward, revealing the mounting bridge, bearing socket, and inner gear drive.
- **Chapter 03: The Mechanism** (`03 / MECHANISM`)
  - Smooth camera dolly and zoom into the drive gear, bearing holder, and nut holder, highlighting precision mechanical tolerances.
- **Chapter 04: The Perspective** (`04 / PERSPECTIVE`)
  - Re-frames the camera stand, demonstrating the single-piece arm geometry and optical angle.
- **Chapter 05: The Complete Architecture** (`05 / EXPLODED VIEW`)
  - Full technical exploded view where each component floats at separate vertical and radial elevations, showcasing the complete mechanical assembly.
- **Back to the Whole (`06`)**
  - Smoothly re-converges all components back into the solid assembled state before transitioning to the component index.

---

### 3. 🏷️ Real-Time 3D Coordinate Annotation Labels
- Floating HUD-style labels (`part-label`) calculate real-time world-to-screen matrix transformations (`project(camera)`).
- Dynamically track individual parts as they move and rotate across chapters, smoothly fading in and out at key narrative moments.

---

### 4. 🎛️ Interactive Controls & Viewport Tools
- **Ambient Rotation Toggle**: Pause or resume continuous turntable rotation at any point in the scroll timeline.
- **Reset View Button**: Smoothly returns the page and 3D camera back to the hero assembled position.
- **Chapter Navigation Dots**: Quick jump between all 5 story stages with accessible ARIA state indicators.
- **Reduced Motion Support**: Automatically respects `prefers-reduced-motion: reduce`, disabling ambient spin and snapping transitions for accessibility.

---

### 5. 📋 Expanded Technical Component Index
- Clean 4-column responsive data table (`NO.`, `COMPONENT`, `DESIGN NOTE`, `SOURCE`).
- Documents all printed parts, hardware, bearings, actuator, and optics.
- **Instant STL Download**: Includes a one-click download for `source-parts.zip` containing all original STL files and print notes.

---

### 6. 📱 Responsive & Mobile-Optimized Design
- Fully responsive across desktop (1920×1080), tablet, and mobile (390×844 iPhone size).
- Mobile viewport adapts the WebGL canvas, collapses the header navigation into quick jump links, and reformats the specification table into an overflow-free card grid.
- Dynamic DPR clamping (`Math.min(devicePixelRatio, 2)`) to ensure 60fps performance across low-power and high-DPI displays.

---

### 7. ☁️ Cloud Deployment Configuration
- Includes [`render.yaml`](./render.yaml) blueprint for 1-click free static site deployment on [Render](https://render.com).
- Optimized Vite production build pipeline generating minified HTML, CSS, and JS in `dist/`.

---

## 📂 Project Directory Structure

```text
scanner-site/
├── index.html              # Semantic HTML5 markup, story chapters, & component table
├── package.json            # Dependencies (Three.js, GSAP, Vite)
├── render.yaml             # Render static site deployment blueprint
├── push-to-github.mjs      # Automated Git synchronization script
├── public/
│   ├── favicon.svg         # SVG wordmark favicon
│   └── models/
│       ├── frame.glb           # 3D Main frame geometry
│       ├── top_plate.glb       # 3D Top plate / platter geometry
│       ├── small_gear.glb      # 3D Drive gear geometry
│       ├── bearing_holder.glb  # 3D Bearing holder geometry
│       ├── nut_holder.glb      # 3D Nut holder geometry
│       ├── camera_stand.glb    # 3D Camera stand geometry
│       ├── assembly-poster.png # High-res static fallback render
│       ├── manifest.json       # Asset dimensions, bounds & transforms
│       └── source-parts.zip    # Original 8 STL files + printing instructions
├── src/
│   ├── main.js             # Three.js scene, GSAP scroll timeline & lighting
│   ├── hardware-models.js  # Procedural 3D models for bearings, fasteners & camera
│   └── style.css           # Vanilla CSS design system, typography & responsiveness
└── tools/
    ├── prepare_models.py   # Blender automated mesh optimization & GLB export
    └── render_assembly.py  # Blender transparent poster generation
```

---

## 💻 Getting Started Locally

### Prerequisites
- Node.js 20.x or 22.x
- npm

### 1. Clone the repository
```bash
git clone https://github.com/kushaldoshi26/3DSCANNER.git
cd 3DSCANNER
```

### 2. Install dependencies
```bash
npm install
```

### 3. Start local development server
```bash
npm run dev
```
Open **`http://127.0.0.1:5173/`** in your browser.

### 4. Build for production
```bash
npm run build
npm run preview
```
Production assets are generated in the `dist/` directory.

---

## 🛠️ Technologies Used

- **Three.js** (`0.186.1`) — WebGL 3D rendering, PBR materials, shadows, and camera projection.
- **GSAP** (`3.15.0`) & **ScrollTrigger** — Smooth scrubbed scroll animations and chapter tracking.
- **Vite** (`8.3.4`) — Lightning-fast development server and optimized rollup bundling.
- **Vanilla CSS** — Custom responsive design system with CSS custom properties, grid, and fluid typography.
- **Blender 5.2.2 LTS** — Automated mesh decimation (reduced from 442,114 to 68,130 triangles).

---

## 📄 License & Credits

- Original 3D printable STL geometry based on the open-source 3D Scanner design.
- Interactive WebGL showcase by [Kushal Doshi](https://github.com/kushaldoshi26).
