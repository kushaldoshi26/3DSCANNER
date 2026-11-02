# Portrait fidelity check - 29 October 2026

## What changed
- Automatic TripoSR source-photo overlay is off. It mapped 2D detail onto mismatched facial geometry and caused visible patches.
- Native Hunyuan texture is the default. Photo overlay remains explicitly experimental and opt-in.
- Mesh smoothing defaults off. Full generated geometry is retained through the texture stage; final cleanup removed four degenerate faces in this run.
- Hunyuan defaults to a front-facing viewer and gallery preview.
- Worker progress now handles JSON interleaved with tqdm, including surface-decoding progress.
- The AI Lab identifies its training as depth/normals/mask only, not Hunyuan or facial-expression training.

## Real application test
Source: C:/Users/Jolly InfoTech/Downloads/IMG_0030.jpg
Job: c776baa8
Gallery: Portrait - Hunyuan Ultra native (29 Oct)
Requested and used: Hunyuan3D 2.0; no fallback.
Ultra: 50 sampling steps, octree 512, seed 42.
Elapsed: 406.13 seconds (6 minutes 46 seconds).
Final: 561,956 vertices, 1,123,932 faces; one connected component, watertight, zero non-manifold edges.
Self-intersections were not tested; scale is not calibrated. Mesh checks are not an accuracy score.
Exports: data/jobs/c776baa8/output/final_model.glb, .stl, .obj, .ply.
Weights remain pretrained. No fine-tuning was performed or claimed.

## Visual result
The native face avoids the previous hard photo-overlay patches. The smile is present, but facial identity/expression and depth are approximate. Fine wrist threads, the watch, teeth and exact clothing motifs are still inaccurate. The back is inferred. More triangles alone do not restore those details.

## Training readiness
- merged_v4: synthetic-only depth candidate; test MAE 0.132077, mask IoU 0.980318, not activated.
- domain_v5: rejected regression, not activated.
- THuman2 imported archive: one textured sample (0525), already used in custom training.
- THuman4 imported archive: documentation/demos; full subject downloads incomplete.
- Existing RGB/depth renders train a different model, not the Hunyuan shape/paint weights.

## Next useful model work
1. Establish held-out portrait tests with original front/side photos and, where available, reference scans. Split by identity, not rendered image.
2. Evaluate a dedicated expression-aware face reconstruction module on face crops, including mouth/eye landmarks and profile depth, before trying to fuse it into the body mesh. The generic generator has no measured identity guarantee.
3. Use provider-authorized high-resolution face scans/expressions and texture data appropriate to that model's training recipe. Do not label generated backs or synthetic guesses as real supervision.
4. Train a separate candidate only when the model, data/license and GPU-memory requirements are verified. Promote only after held-out face/texture improvement without hand/body regressions.

References reviewed:
https://github.com/yfeng95/DECA (face geometry/expression module and training requirements; research license applies)
https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1 (separate newer model/training code; official texture inference requirement 21GB VRAM, above this GPU's 16GB without an additional memory strategy)
https://github.com/VAST-AI-Research/TripoSR (open-source TripoSR is a separate baseline)

## Personal test
Open http://localhost:8000/ and Workspace. Choose Hunyuan3D, Ultra, smoothing off, photo overlay off. Upload one clear image and Generate. Inspect the result from front, side and back; use Shaded to assess geometry independently of texture. Review actual engine in Result details. Do not run another GPU training job during generation.
