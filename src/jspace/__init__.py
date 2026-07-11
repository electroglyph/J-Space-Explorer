"""J-Space Explorer.

Tools for exploring the J-space (Jacobian lens) and logit lens of local
safetensors language models, following the Transformer Circuits "Verbalizable
Representations Form a Global Workspace" methodology.
"""
from __future__ import annotations

__version__ = "0.1.0"

from .config import AppConfig, LensConfig, ModelConfig

__all__ = ["AppConfig", "LensConfig", "ModelConfig", "__version__"]
