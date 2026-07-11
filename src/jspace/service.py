"""Model service singleton shared by the web app and CLI.

Loads the model once (it is large) and holds the lens engine + J-space
decomposer. Thread-safe lazy init so the FastAPI app can start instantly and
load the model on first request (or eagerly at startup).
"""
from __future__ import annotations

import logging
import threading
from typing import Optional

from .config import AppConfig
from .jspace import JSpaceDecomposer
from .lens import LensEngine
from .model import LensModel

logger = logging.getLogger(__name__)


class ModelService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._model: Optional[LensModel] = None
        self._engine: Optional[LensEngine] = None
        self._decomposer: Optional[JSpaceDecomposer] = None
        self._translator = None
        self._lock = threading.Lock()
        self._loading = False
        self._error: Optional[str] = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def status(self) -> dict:
        info = None
        if self._model is not None:
            i = self._model.info
            info = {
                "architecture": i.architecture,
                "num_layers": i.num_layers,
                "hidden_size": i.hidden_size,
                "vocab_size": i.vocab_size,
                "dtype": str(i.dtype),
            }
        return {
            "loaded": self.is_loaded,
            "loading": self._loading,
            "error": self._error,
            "model_path": self.config.model.path,
            "info": info,
            "fitted_layers": (
                sorted(self._engine._jacobian.keys())
                if self._engine is not None else []
            ),
        }

    def ensure_loaded(self) -> LensModel:
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            self._loading = True
            self._error = None
            try:
                self._model = LensModel.load(self.config.model)
                self._engine = LensEngine(self._model, self.config.lens)
                self._decomposer = JSpaceDecomposer(
                    self._model, self._engine, self.config.lens
                )
                from .translate import Translator
                self._translator = Translator(self._model)
            except Exception as exc:  # surfaced to the client
                self._error = f"{type(exc).__name__}: {exc}"
                logger.exception("Model load failed")
                raise
            finally:
                self._loading = False
        return self._model

    @property
    def engine(self) -> LensEngine:
        self.ensure_loaded()
        assert self._engine is not None
        return self._engine

    @property
    def decomposer(self) -> JSpaceDecomposer:
        self.ensure_loaded()
        assert self._decomposer is not None
        return self._decomposer

    @property
    def translator(self):
        self.ensure_loaded()
        assert self._translator is not None
        return self._translator
