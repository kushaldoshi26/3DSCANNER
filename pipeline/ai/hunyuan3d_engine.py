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
        return probe_module("hy3dgen.shapegen", "Hunyuan3DDiTFlowMatchingPipeline")

    def generate(
        self,
        image_path: str,
        output_stl: str,
        output_glb: str,
        texture: bool = True,
        quality: str = "high",
        progress_cb=None,
    ) -> dict:
        self._progress(progress_cb, "AI reconstruction", 40, "Loading Hunyuan3D 2.0 pipelines...")
        from pipeline.hunyuan3d_inference import generate_hunyuan3d_mesh

        self._progress(progress_cb, "AI reconstruction", 60, "Hunyuan3D shape + paint synthesis...")
        try:
            generate_hunyuan3d_mesh(image_path, output_stl, output_glb, paint_texture=texture)
        except Exception as e:
            raise RuntimeError(f"Hunyuan3D generation failed: {e}")

        self._progress(progress_cb, "AI reconstruction", 90, "Hunyuan3D mesh exported.")
        return {"engine_used": self.key, "degraded": False}
