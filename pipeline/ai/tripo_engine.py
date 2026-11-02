"""
TripoSR (Stability AI) engine wrapper — fast fallback.

Delegates to the existing, tested `pipeline.triposr_inference` implementation
and exposes the uniform `EngineBase` contract used by `engine_manager`.
"""

from pipeline.ai.engine_base import EngineBase, probe_module

_QUALITY_RESOLUTION = {
    "fast": 256,
    "high": 384,
    "ultra": 512,
    "medium": 384,
    "low": 256,
}


class TripoEngine(EngineBase):
    key = "tripo"
    label = "TripoSR (Stability AI)"
    detail = "Fast tri-plane LRM. ~1s inference, solid 360° geometry."
    fallbacks = ["ai_depth"]

    def is_available(self) -> bool:
        return probe_module("tsr.system", "TSR")

    def generate(
        self,
        image_path: str,
        output_stl: str,
        output_glb: str,
        texture: bool = True,
        quality: str = "high",
        progress_cb=None,
    ) -> dict:
        from pipeline.triposr_inference import generate_triposr_3d_mesh

        resolution = _QUALITY_RESOLUTION.get(quality, 384)
        self._progress(progress_cb, "AI reconstruction", 40, "Loading TripoSR model...")
        self._progress(progress_cb, "AI reconstruction", 60, f"TripoSR tri-plane inference (res={resolution})...")
        try:
            generate_triposr_3d_mesh(
                image_path,
                output_stl,
                output_glb,
                resolution=resolution,
            )
        except Exception as e:
            raise RuntimeError(f"TripoSR generation failed: {e}")

        self._progress(progress_cb, "AI reconstruction", 90, "TripoSR mesh exported.")
        return {"engine_used": self.key, "degraded": False}
