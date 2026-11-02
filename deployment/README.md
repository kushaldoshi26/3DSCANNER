# Deployment preparation (not yet cloud-validated)

GitHub contains the application, not a copy of the developer workstation.

| Asset | Deployment treatment |
|---|---|
| API, UI, pipeline, homepage GLBs | Included in Git |
| Hunyuan shape/delight/paint weights | Download from official provider into persistent storage |
| Hunyuan engine code and CUDA extensions | Install/build on target GPU host using official instructions |
| Hugging Face/rembg/other auxiliary caches | Provision on target host before offline inference; test full generation |
| Training datasets, rejected custom candidates | Not required for pretrained Hunyuan inference |
| Existing jobs and account database | Optional private migration, never publish in Git |
| OAuth secrets | Hosting secret/environment settings only |
| Backups, Windows packages/toolchains | Do not deploy |

## Model provisioning
Review https://github.com/Tencent-Hunyuan/Hunyuan3D-2 and its LICENSE.
Install huggingface_hub in the target environment. Run `python deployment/download_models.py --help`.
Use `--revision` with the official model repository's 40-character commit SHA and `--accept-license` after reviewing the license. The downloader is resumable and pins the snapshot; it has not downloaded weights again on the local PC.
Set HY3DGEN_MODELS to a persistent models directory and SCANNER_HUNYUAN_SOURCE to the installed official source directory. Build custom_rasterizer and the other official texture dependencies for the target OS/CUDA. Existing Windows binaries cannot be copied to Linux.

## Hosting
The current generator requires CUDA. Render's documented standard compute plans describe CPU/RAM; do not assume a normal Render web service can run this generator. Use a verified CUDA GPU host for the full application, or implement and test a separate GPU worker service before splitting the frontend/API onto Render. That remote-worker integration is not implemented here.
Start on a configured host with `python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1` (substitute the hosting PORT). Keep data/ on persistent storage. Multiple workers are not safe with current in-memory session/job coordination.

## Remaining release gates
- Resolve and lock Linux-compatible Python dependencies: root requirements.txt still includes local hardware dependencies and an invalid pyserial==3.50 pin; it is not a tested cloud build specification.
- Run one complete textured image generation after provisioning all auxiliary assets; workers currently use HF_HUB_OFFLINE=1.
- Verify login, OAuth callbacks, job authorization, HTTPS, rate limits and storage persistence before public exposure.
- Test restart/recovery; in-memory login sessions are lost on restart.
- Do not claim model accuracy from installation success. Face/watch/clothing detail remains approximate.

No paid hosting, dataset publication, credential upload, or production deployment is performed by these files.
