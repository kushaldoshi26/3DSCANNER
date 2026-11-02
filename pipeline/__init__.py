"""Project-local model caches: keep scanner assets together on one drive."""
import os
from pathlib import Path
_root=Path(__file__).resolve().parent.parent
os.environ.setdefault('HF_HOME',str(_root/'data/cache/huggingface'))
os.environ.setdefault('HF_HUB_CACHE',str(_root/'data/cache/huggingface/hub'))
os.environ.setdefault('U2NET_HOME',str(_root/'data/cache/u2net'))
os.environ.setdefault('HY3DGEN_MODELS',str(_root/'engines/models'))
os.environ.setdefault('SCANNER_HUNYUAN_SOURCE',str(_root/'engines/Hunyuan3D-2-main'))
