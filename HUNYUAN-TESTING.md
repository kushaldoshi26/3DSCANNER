# Hunyuan local benchmark (26 October 2026)

These commands run pretrained inference, not fine-tuning. Do not run both simultaneously.

## Portrait texture test

    python D:\3dSCANNER\pipeline\test_hunyuan_texture.py
    python D:\3dSCANNER\pipeline\watch_texture_setup.py

The verified portrait preview is in the gallery as `Hunyuan Test - IMG_0030`.
Its texture is 2048 x 2048, with a 100,000-face preview mesh.
Unseen surfaces are inferred. This preview has two non-manifold edges in the
welded STL and must not be advertised as print-ready.

## Statue shape and texture test

    python D:\3dSCANNER\pipeline\test_hunyuan_statue.py

Progress log:

    Get-Content D:\3dSCANNER\data\software_checks\hunyuan_texture_setup\statue-test.log -Tail 10 -Wait

Status: `data/software_checks/hunyuan_statue/status.json`.
Ctrl+C stops the log viewer only. Large-mesh UV preparation can be CPU-heavy.

## Memory and source handling

The paint pipelines are initialized on CPU and then use GPU model offload.
The large custom UNet loads through a meta initialization and memory-mapped
strict state assignment. The Windows shape loader uses the NumPy safetensors
reader to avoid the observed PyTorch storage access violation.
Downloaded weights remain pretrained; no trained checkpoint was promoted.
Local source patches are required and can be overwritten by upstream reinstall.

## What qualifies as progress

A completed export proves that inference ran. It does not prove source-image
accuracy, unseen-back correctness, printability or real-world scale. Inspect
front/side/back renders and geometry reports independently. The custom depth
candidate is a separate network and does not fine-tune Hunyuan or TripoSR.