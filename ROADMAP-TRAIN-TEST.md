# Current project status - 2 November 2026

This section supersedes the historical roadmap below. No cloud deployment has been performed.

## Verified implementation
- API, workspace/viewer, gallery, account UI, OAuth flows, training utilities and hardware controller source are tracked in Git.
- Native Hunyuan textured portrait generation succeeded on 29 October: job c776baa8, 406.13 seconds. See PORTRAIT-QUALITY-20261029.md for evidence and limitations.
- Latest checks: 49 focused Python tests and JavaScript account/gallery/generation/point-view checks passed; six runtime tests passed after the portability change.
- merged_v4 completed 20 epochs (best 12), synthetic-only and not activated. domain_v5 completed 11 epochs (best 3), regressed and was not activated. These train depth/mask models, not Hunyuan.

## Remaining work in priority order
1. Establish a fixed unseen real-photo evaluation set; compare face expression/profile, hands, accessories and fabric detail separately. Current generation is approximate, with inferred backs.
2. Verify model-specific training data and a feasible training recipe before new fine-tuning. THuman2 currently supplies one imported textured sample; THuman4 full training data is not ready. Do not merge incompatible datasets blindly.
3. Evaluate a face/expression refinement candidate and texture preservation against the same baseline, then train only if justified. Hunyuan weights remain pretrained.
4. Finish physical camera mount and run actual ESP32 camera/turntable capture, calibration and reconstruction tests. Controller source alone does not establish scan accuracy.
5. Configure Google/Facebook provider credentials privately and test real callbacks. Backend implementation is not proof of successful provider login.
6. Check installation reproducibility: requirements.txt still has pyserial==3.50 and local hardware dependencies; resolve versions in an isolated environment before changing the working installation. Cloud/Linux installation is not validated.
7. Perform a fresh live UI/API test after starting the local server, including restart/recovery, exports and job failure handling. The recent API probe timed out.

## Repository scope
GitHub branch: scanner-local-integration. The GitHub main branch has unrelated website history; it has not been overwritten or merged. Source, tests and required homepage model assets are published. Private job/account data, datasets, model weights, local engine installations and backups remain local. deployment/ contains preparation instructions only; no hosting was started.

---

# Historical roadmap (23 October; retained for context)

# 3D SCANNER: roadmap, training and testing

## Current state

- Active project: `D:\3dSCANNER`.
- Merged depth/normal/mask candidate: training completed, 28 epochs, selected epoch 20.
- Data: 1,145 training views, 188 validation views, 186 held-out synthetic test views. Object identities do not overlap across splits.
- Final synthetic depth MAE: 0.132348, mask IoU: 0.983190. These do not establish real-photo or millimetre accuracy.
- Hunyuan shape weights: downloaded and safetensors structure verified. First portrait run failed in upstream CPU-offload helper; that call was removed for the 16GB GPU and the test restarted.
- Hunyuan texture stage: CUDA rasterizer/compiler and actual texture inference remain unverified.
- THuman 2 ZIP contains one textured example scan, now in training. Its old diagnostic score is no longer an independent test. THuman 4 ZIP contains only repository documentation/demos; full subject archives are not present.

## Order of work

1. Complete Hunyuan shape test. Check foreground similarity, upright orientation, connected parts, fingers, face, and side/back shape. Keep identical source image and seed for comparisons.
2. Set up and test native texture generation. Check front/back colour consistency, UV seams, fabric scale and skin detail. Compare with the working TripoSR export. A repeated inferred cloth patch is not evidence of an observed back.
3. Build a fixed evaluation set of unseen objects and people, with permission to use the images. Prefer front/side/back photographs or reference scans. Keep identities and related geometry out of training.
4. Expand training data only after establishing that evaluation baseline. Keep original textured assets and attribution. Render aligned RGB, float depth, normals and masks; use one consistent camera convention. Geometry-only STL/NPZ files cannot be relabelled as photographic supervision.
5. Retrain a separate candidate and compare against the previous checkpoint on the same held-out set. Promote only after better results and no meaningful regressions. Synthetic-only scores are insufficient.
6. Integrate the verified engine into the UI with truthful availability, progress, cancellation/errors and exports. Hardware capture/calibration is a later separate validation step.

The custom candidate predicts visible depth/normals/masks. It does not learn complete textured 360-degree meshes. Hunyuan is a separate pretrained generator; running depth training does not fine-tune Hunyuan. Its outputs must not be fed into unrelated AI training, in accordance with the accepted Tencent licence.

## Locations

| Purpose | Path under `D:\3dSCANNER` |
|---|---|
| Main API | `api\main.py` |
| UI | `static` |
| Original geometry | `data\datasets\*.stl` |
| Prepared geometry | `data\datasets\processed` |
| Textured originals and aligned renders | `data\datasets\photo_v2_run` |
| Imported THuman example | `data\datasets\thuman_reference` |
| Merged manifest | `data\datasets\merged_training_manifest.json` |
| Merged custom candidate | `data\models\merged_v3_candidate` |
| Native Hunyuan source | `engines\Hunyuan3D-2-main` |
| Downloaded Hunyuan weights | `engines\models\tencent\Hunyuan3D-2\hunyuan3d-dit-v2-0` |
| Hunyuan portrait test and logs | `data\software_checks\hunyuan_portrait` |
| Working TripoSR comparison | `data\software_checks\portrait_complete` |

## PowerShell commands

```powershell
$py = "C:\Users\Jolly InfoTech\AppData\Local\Programs\Python\Python311\python.exe"
Set-Location D:\3dSCANNER
$env:PYTHONIOENCODING = "utf-8"
$env:OMP_NUM_THREADS = "4"
```

Check the current Hunyuan test; do not start a second GPU job while it is running:

```powershell
Get-Content data\software_checks\hunyuan_portrait\status.json
Get-Content data\software_checks\hunyuan_portrait\retry-errors.log -Tail 20
```

Re-run the Hunyuan shape-only test when no generation is active:

```powershell
& $py pipeline/test_hunyuan_portrait.py
```

Training is already complete. To intentionally retrain the merged custom depth candidate, back up its existing output first:

```powershell
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
Copy-Item -LiteralPath data\models\merged_v3_candidate -Destination "data\models\merged_v3_backup-$stamp" -Recurse
& $py pipeline/merge_and_train.py
```

This performs a fresh fine-tuning run from `photo_v2_textured_candidate/best.pth`; it does not resume the latest merged epoch or train Hunyuan. The script evaluates held-out synthetic test objects automatically and writes `status.json`, `history.json`, `best.pth` and `best.evaluation.json` in the candidate folder. Repeated inspection of this test set is development feedback, not a new untouched benchmark; add a fresh final benchmark before a release claim.

```powershell
Get-Content data\models\merged_v3_candidate\status.json
Get-Content data\models\merged_v3_candidate\best.evaluation.json
& $py -m pytest tests/test_volume_training.py tests/test_source_detail.py tests/test_generation_regressions.py tests/test_generation_api.py -q -p no:cacheprovider
node tests/test_source_preview.js
node tests/test_generation_form.js
& $py api/main.py
```

Open `http://localhost:8000/`, upload the test photo in Workspace, choose the engine that is actually ready, generate, orbit around the result, inspect points/texture, and export GLB. Triangle counts or a watertight flag alone do not establish visual fidelity or real-world dimensions.
