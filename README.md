# FORM — 3D scanner showcase

A scroll-driven product website built with Three.js, GSAP ScrollTrigger, and Vite. It uses the supplied scanner STL geometry for assembly, exploded views, and component close-ups. No backend is required.

## Run locally

Use Node.js 20.19+ in the 20.x line, or Node.js 22.12+.

Clone or download this repository, then open a terminal in its root folder (the folder containing `package.json`).

```powershell
npm ci
npm run dev
```

Open the local URL printed by Vite. The development server is bound to `127.0.0.1`.

```powershell
npm run build
npm run preview
```

The production files are written to `dist/`. Serve the build through HTTP; opening `index.html` directly from the filesystem will not load the module-based 3D scene correctly.

## Project files

| Path | Purpose |
| --- | --- |
| `index.html` | Page sections, navigation, component details, and source download |
| `src/main.js` | Three.js scene, materials, assembly positions, scroll shots, and fallback behavior |
| `src/style.css` | Responsive layout and visual styling |
| `public/models/*.glb` | Six centered, optimized component models |
| `public/models/assembly-poster.png` | Transparent Blender render used as the static fallback |
| `public/models/source-parts.zip` | All eight original STL files and the printing notes |
| `public/models/manifest.json` | Geometry counts, dimensions, source paths, and inferred transforms |
| `tools/prepare_models.py` | Blender import, optimization, GLB export, and inspection renders |
| `tools/render_assembly.py` | Blender assembly preview and transparent poster generation |
| `model-reports/` | Generated locally: geometry report, contact sheet, and editable Blender scenes; excluded from Git |

## Blender assets

Blender **5.2.2 LTS** was used. Blender is needed only to regenerate assets; the website runs using the existing GLBs.

The scripts expect a folder named `3D scanner 3D printed parts` next to the repository folder. After a fresh clone, restore that folder from the included archive before running Blender:

```powershell
Expand-Archive -LiteralPath '.\public\models\source-parts.zip' -DestinationPath '..'
```

For a Microsoft Store Blender installation, use its execution alias:

```powershell
$blenderExe = Join-Path $env:LOCALAPPDATA 'Microsoft\WindowsApps\blender-launcher.exe'
$prepareScript = (Resolve-Path '.\tools\prepare_models.py').Path
$renderScript = (Resolve-Path '.\tools\render_assembly.py').Path
& $blenderExe --background --factory-startup --python $prepareScript
& $blenderExe --background --factory-startup --python $renderScript
```

Run these commands from the repository root and let preparation finish before rendering. With another Blender installation, replace `$blenderExe` with its executable path.

The scripts resolve project paths relative to their own files. They read the originals and write generated files only inside the repository. The following files are generated locally and are not included in a clone:

- `model-reports/scanner-assets.blend` contains the six centered meshes at the original numerical scale.
- `model-reports/scanner-assembly-preview.blend` contains the lit presentation assembly.
- `model-reports/geometry-report.json` records the asset conversion details.

The selected meshes were reduced from **442,114 to 68,130 triangles**; their GLBs total approximately **2.03 MB**. The one-piece camera stand is used in the scene. Its two printable halves remain available in the source ZIP as alternative print variants.

## Assembly assumptions

STL files do not encode units. Dimensions are treated as millimetres based on the supplied print instructions. Models retain their relative dimensions and are individually centered; glTF uses Y-up coordinates. The website applies a common scale of `0.023`.

The current displayed part arrangement is inferred from geometry and presents the supplied printed structure. The project's [Instructables reference](https://www.instructables.com/3D-Printable-3D-Scanner-Using-Intel-Realsense-D435/) has since been reviewed to plan the camera, motor, electronics, connecting rail, and correct mechanical motion. Those additions are planned and are not implemented in this version. The animation does not represent a validated physical assembly or actual scan results.

## Asset checks

All six GLB headers, embedded buffers, named nodes, and triangle counts were checked against the manifest. Required model, poster, ZIP, and Blender paths exist. The ZIP contains eight STLs and one text file, and every archived file matches its corresponding current source file by SHA-256. No earlier checksum baseline was available for a historical comparison.
