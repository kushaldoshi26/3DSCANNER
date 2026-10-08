"""
TRELLIS (Microsoft) engine wrapper.

Delegates to the existing, tested `pipeline.trellis_inference` implementation
and exposes the uniform `EngineBase` contract used by `engine_manager`.
"""

from pipeline.ai.engine_base import EngineBase, probe_module


class TrellisEngine(EngineBase):
    key = "trellis"
    label = "TRELLIS (Microsoft)"
    detail = "Structured LATent (SLAT). Highest geometric fidelity."
    fallbacks = ["hunyuan3d", "tripo"]

    def is_available(self) -> bool:
        return probe_module("trellis.pipelines", "TrellisImageTo3DPipeline")

    def generate(
        self,
        image_path: str,
        output_stl: str,
        output_glb: str,
        texture: bool = True,
        quality: str = "high",
        progress_cb=None,
    ) -> dict:
        self._progress(progress_cb, "AI reconstruction", 40, f"Loading TRELLIS on GPU...")
        from pipeline.trellis_inference import generate_trellis_3d_mesh

        self._progress(progress_cb, "AI reconstruction", 60, "Running TRELLIS SLAT inference...")
        try:
            generate_trellis_3d_mesh(image_path, output_stl, output_glb)
        except Exception as e:
            raise RuntimeError(f"TRELLIS generation failed: {e}")

        self._progress(progress_cb, "AI reconstruction", 90, "TRELLIS mesh exported.")
        return {"engine_used": self.key, "degraded": False}
