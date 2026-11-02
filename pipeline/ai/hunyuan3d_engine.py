"""
Hunyuan3D 2.0 (Tencent) engine wrapper.

Delegates to the existing, tested `pipeline.hunyuan3d_inference` implementation
and exposes the uniform `EngineBase` contract used by `engine_manager`.
"""

from pipeline.ai.engine_base import EngineBase, probe_module


class Hunyuan3DEngine(EngineBase):
    key = "hunyuan3d"
    label = "Hunyuan3D 2.0 (Tencent)"
    detail = "DiT flow-matching shape + Paint texture synthesis."
    fallbacks = ["trellis", "tripo"]

    def is_available(self) -> bool:
        from pipeline.hunyuan3d_inference import _native_imports
        _native_imports()
        from pathlib import Path
        import os,importlib.util
        models=Path(os.environ['HY3DGEN_MODELS'])/'tencent/Hunyuan3D-2'
        required=['hunyuan3d-dit-v2-0/model.fp16.safetensors','hunyuan3d-delight-v2-0/unet/diffusion_pytorch_model.safetensors','hunyuan3d-paint-v2-0-turbo/unet/diffusion_pytorch_model.bin']
        return all((models/p).is_file() for p in required) and importlib.util.find_spec('custom_rasterizer_kernel') is not None and probe_module('hy3dgen.shapegen','Hunyuan3DDiTFlowMatchingPipeline')

    def generate(
        self,
        image_path: str,
        output_stl: str,
        output_glb: str,
        texture: bool = True,
        quality: str = "high",
        progress_cb=None,
    ) -> dict:
        self._progress(progress_cb, "AI reconstruction", 1, "Loading Hunyuan3D 2.0 pipelines...")
        from pipeline.hunyuan_runtime import run_hunyuan_worker

        try:
            metadata=run_hunyuan_worker(image_path,output_stl,output_glb,texture=texture,quality=quality,progress_cb=progress_cb)
        except Exception as e:
            raise RuntimeError(f"Hunyuan3D generation failed: {e}")

        self._progress(progress_cb, "AI reconstruction", 90, "Hunyuan3D mesh exported.")
        return {"engine_used": self.key, "degraded": False, "generation":metadata}
