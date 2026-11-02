# Scanner project location

The active project is consolidated under `D:\3dSCANNER`.

| Content | Location |
|---|---|
| API, UI, pipeline and hardware source | `D:\3dSCANNER` |
| Training datasets and manifests | `D:\3dSCANNER\data\datasets` |
| Trained custom checkpoints | `D:\3dSCANNER\data\models` |
| Pretrained runtime caches | `D:\3dSCANNER\data\cache` |
| Official Hunyuan source | `D:\3dSCANNER\engines\Hunyuan3D-2-main` |
| Hunyuan model weights and resumable download | `D:\3dSCANNER\engines\models` |
| Generation checks and results | `D:\3dSCANNER\data\software_checks` |

F: copies and previous C: caches are preserved as backups; the updated project paths use D:. Python itself remains the existing Windows installation on C:, just like other installed development tools.

## Run

```powershell
$py = "C:\Users\Jolly InfoTech\AppData\Local\Programs\Python\Python311\python.exe"
Set-Location D:\3dSCANNER
& $py api/main.py
```

Model download resumes using `pipeline/download_hunyuan_ranges.py`. The setup helper is `pipeline/finish_hunyuan_setup.py`; do not start duplicate copies while it is already running.

All 31 reconstruction regression tests and both JavaScript generation/point-cloud checks passed after the path updates. This verifies existing scanner behaviour, not a completed Hunyuan generation: its weights and actual test result have separate status files.
