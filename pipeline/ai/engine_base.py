"""
Shared interface for single-image 3D engines.

Every engine (TRELLIS, Hunyuan3D, TripoSR) exposes the same uniform
`EngineBase` contract so the `engine_manager` can swap, probe availability,
and walk a fallback chain without knowing engine-specific details.
"""

from abc import ABC, abstractmethod
from typing import Callable, Dict, Any, Optional

# (stage_name, percent, message)
ProgressCallback = Callable[[str, int, str], None]


class EngineBase(ABC):
    # Populated by concrete engines
    key: str = ""
    label: str = ""
    detail: str = ""
    fallbacks: list = []

    @abstractmethod
    def is_available(self) -> bool:
        """Best-effort check that the engine's native package is importable."""
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        image_path: str,
        output_stl: str,
        output_glb: str,
        texture: bool = True,
        quality: str = "high",
        progress_cb: Optional[ProgressCallback] = None,
    ) -> Dict[str, Any]:
        """
        Generates a 3D model from a single image.

        Returns a dict with at least {'engine_used': key, 'degraded': bool}.
        Raises RuntimeError on hard failure so the manager can fall back.
        """
        raise NotImplementedError

    def _progress(
        self,
        cb: Optional[ProgressCallback],
        stage: str,
        percent: int,
        message: str,
    ) -> None:
        if cb:
            cb(stage, percent, message)


def probe_module(module_name: str, attr: Optional[str] = None) -> bool:
    """Returns True if `module_name` (and optional `attr`) imports cleanly."""
    import importlib
    try:
        mod = importlib.import_module(module_name)
        return attr is None or hasattr(mod, attr)
    except Exception:
        return False
