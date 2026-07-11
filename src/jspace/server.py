"""FastAPI backend for the J-Space Explorer."""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import AppConfig
from .service import ModelService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "web" / "static"


class LensRequest(BaseModel):
    prompt: str = Field(..., description="Prompt to run the lens on")
    method: str = Field("logit", description="'logit' or 'jacobian'")
    layers: Optional[list[int]] = None
    positions: Optional[list[int]] = None
    top_k: Optional[int] = None
    use_chat_template: bool = True
    corpus: Optional[list[str]] = None


class BranchRequest(BaseModel):
    """Replace the token at ``position`` with ``branch_token_id`` and re-run the
    lens on the modified context — to see how a change far back in the context
    reshapes the model's downstream thinking."""
    prompt: str = ""
    messages: Optional[list[dict]] = None
    # One or more position->token edits, applied to the freshly-built input_ids
    # of prompt/messages. Accumulating edits (rather than re-tokenizing decoded
    # text) keeps the branched context exact.
    edits: list[dict] = []
    method: str = "logit"
    layers: Optional[list[int]] = None
    top_k: Optional[int] = None
    use_chat_template: bool = True
    corpus: Optional[list[str]] = None
    at_generation_point: bool = True


class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: int = 128
    use_chat_template: bool = True


class StreamGenerateRequest(BaseModel):
    prompt: str = ""
    messages: Optional[list[dict]] = None
    method: str = "logit"           # 'logit' | 'jacobian'
    layers: Optional[list[int]] = None
    top_k: int = 6
    max_new_tokens: int = 200
    use_chat_template: bool = True
    temperature: float = 0.0        # 0 = greedy
    top_p: Optional[float] = None
    sampling_top_k: Optional[int] = None
    min_p: Optional[float] = None


class DecomposeRequest(BaseModel):
    prompt: str
    layer: int
    position: int
    use_chat_template: bool = True
    corpus: Optional[list[str]] = None
    # In chat mode the lens ran on a conversation; pass it so decompose
    # tokenizes identically (positions line up with the grid).
    messages: Optional[list[dict]] = None
    at_generation_point: bool = True


class FitRequest(BaseModel):
    layers: Optional[list[int]] = None
    # Must match LensRequest/DecomposeRequest (True), or the fit signature
    # differs from the apply signature and operators get silently refit.
    use_chat_template: bool = True
    corpus: Optional[list[str]] = None


class TranslateRequest(BaseModel):
    tokens: list[str]


class TokenizeRequest(BaseModel):
    text: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    min_p: Optional[float] = None


class ChatLensRequest(BaseModel):
    messages: list[ChatMessage]
    method: str = "logit"        # 'logit' or 'jacobian'
    layers: Optional[list[int]] = None
    positions: Optional[list[int]] = None
    top_k: Optional[int] = None
    corpus: Optional[list[str]] = None
    # Run the lens at the assistant generation point (append the assistant
    # header) so the readout shows what the model is poised to say next.
    at_generation_point: bool = True


class Intervention(BaseModel):
    layer: int
    token_id: int
    mode: str = "steer"          # 'steer' | 'ablate' | 'swap'
    strength: float = 0.3
    swap_to: Optional[int] = None


class InterveneRequest(BaseModel):
    prompt: Optional[str] = None
    messages: Optional[list[ChatMessage]] = None
    interventions: list[Intervention] = []
    max_new_tokens: int = 200
    temperature: float = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = None
    min_p: Optional[float] = None
    include_baseline: bool = True
    use_chat_template: bool = True


def create_app(config: Optional[AppConfig] = None,
               service: Optional[ModelService] = None) -> FastAPI:
    config = config or AppConfig.from_env()
    service = service or ModelService(config)
    app = FastAPI(title="J-Space Explorer", version="0.1.0")

    @app.get("/api/status")
    def status():
        return service.status

    @app.post("/api/load")
    def load():
        try:
            service.ensure_loaded()
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))
        return service.status

    @app.post("/api/fit")
    async def fit(req: FitRequest):
        """Fit the Jacobian operators for the requested layers, streaming
        progress via SSE. Safe to call repeatedly: already-fitted layers over
        the same corpus are skipped."""
        from .lens import default_corpus

        async def gen():
            import threading
            engine = service.engine
            loop = asyncio.get_running_loop()
            num_layers = engine.model.info.num_layers
            layers = req.layers or list(range(num_layers + 1))
            layers = sorted({l for l in layers if 0 <= l <= num_layers})
            corpus = req.corpus or default_corpus()
            n_ctx = min(engine.config.jacobian_corpus_size, len(corpus))
            yield _sse({"type": "fit_start", "layers": layers,
                        "corpus_size": n_ctx,
                        "already_fitted": sorted(engine._jacobian.keys())})

            q: asyncio.Queue = asyncio.Queue()

            def emit(ev):
                loop.call_soon_threadsafe(q.put_nowait, ev)

            def worker():
                try:
                    # ensure_fitted is idempotent and thread-safe (locked); it
                    # clears on corpus change, fits only missing layers, and
                    # persists to disk. Progress streams to the SSE queue.
                    engine.ensure_fitted(
                        layers, corpus, req.use_chat_template,
                        progress=lambda d, t: emit(
                            {"type": "fit_progress", "done": d, "total": t}
                        ),
                    )
                    emit({"type": "done",
                          "fitted_layers": sorted(engine._jacobian.keys())})
                except Exception as exc:
                    logger.exception("fit failed")
                    emit({"type": "error", "detail": str(exc)})

            threading.Thread(target=worker, daemon=True).start()
            while True:
                ev = await q.get()
                yield _sse(ev)
                if ev.get("type") in ("done", "error"):
                    break

        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/lens")
    def lens(req: LensRequest):
        try:
            engine = service.engine
            if req.method == "jacobian":
                result = engine.jacobian_lens(
                    req.prompt, layers=req.layers, positions=req.positions,
                    corpus=req.corpus, use_chat_template=req.use_chat_template,
                    top_k=req.top_k,
                )
            else:
                result = engine.logit_lens(
                    req.prompt, layers=req.layers, positions=req.positions,
                    use_chat_template=req.use_chat_template, top_k=req.top_k,
                )
            return result.to_dict()
        except Exception as exc:
            logger.exception("lens failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/branch")
    def branch(req: BranchRequest):
        """Replace the token at ``position`` with ``branch_token_id`` and re-run
        the lens on the modified context.

        The swap happens at the exact input_ids index, so a change far back in
        the context propagates through every downstream position's readout.
        """
        try:
            engine = service.engine
            model = engine.model
            # Rebuild the exact same input_ids the original lens saw.
            if req.messages:
                msgs = [{"role": m.get("role"), "content": m.get("content")}
                        for m in req.messages]
                input_ids = model.build_chat_ids(
                    msgs, add_generation_prompt=req.at_generation_point)
            else:
                input_ids = model.build_input_ids(
                    req.prompt, req.use_chat_template)
            seq_len = input_ids.shape[1]
            branched = input_ids.clone()
            applied = []
            for e in req.edits:
                pos = int(e.get("position", -1))
                tid = int(e.get("branch_token_id", -1))
                if not (0 <= pos < seq_len):
                    raise ValueError(
                        f"position {pos} out of range (0..{seq_len - 1})")
                if not (0 <= tid < model.info.vocab_size):
                    raise ValueError(f"branch_token_id {tid} out of vocab range")
                branched[0, pos] = tid
                applied.append({"position": pos, "branch_token_id": tid,
                                "branch_token": model.decode_token(tid)})
            if not applied:
                raise ValueError("no edits provided")

            if req.method == "jacobian":
                result = engine.jacobian_lens(
                    prompt="", layers=req.layers, corpus=req.corpus,
                    top_k=req.top_k, input_ids=branched)
            else:
                result = engine.logit_lens(
                    prompt="", layers=req.layers, top_k=req.top_k,
                    input_ids=branched)
            out = result.to_dict()
            out["branch"] = {"edits": applied}
            return out
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("branch failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/lens/stream")
    async def lens_stream(req: LensRequest):
        """Stream lens rows layer-by-layer via SSE so the UI fills in live."""
        async def gen():
            try:
                engine = service.engine
                loop = asyncio.get_running_loop()

                # Determine layers up front to stream one at a time.
                input_ids = await loop.run_in_executor(
                    None,
                    lambda: engine.model.build_input_ids(
                        req.prompt, req.use_chat_template
                    ),
                )
                num_layers = engine.model.info.num_layers
                layers = req.layers or list(range(num_layers + 1))
                tokens = engine.model.token_strings(input_ids)
                positions = req.positions or list(range(len(tokens)))
                header = {
                    "type": "header",
                    "tokens": [tokens[p] for p in positions if 0 <= p < len(tokens)],
                    "layers": layers,
                    "method": req.method,
                }
                yield _sse(header)

                for ell in layers:
                    def run_one(ell=ell):
                        if req.method == "jacobian":
                            return engine.jacobian_lens(
                                req.prompt, layers=[ell], positions=positions,
                                corpus=req.corpus,
                                use_chat_template=req.use_chat_template,
                                top_k=req.top_k,
                            )
                        return engine.logit_lens(
                            req.prompt, layers=[ell], positions=positions,
                            use_chat_template=req.use_chat_template,
                            top_k=req.top_k,
                        )
                    result = await loop.run_in_executor(None, run_one)
                    cells = [
                        {
                            "layer": c.layer,
                            "position": c.position,
                            "top": [
                                {"token_id": t.token_id, "token": t.token,
                                 "score": t.score}
                                for t in c.top
                            ],
                        }
                        for c in result.cells
                    ]
                    yield _sse({"type": "layer", "layer": ell, "cells": cells})
                yield _sse({"type": "done"})
            except Exception as exc:
                logger.exception("lens stream failed")
                yield _sse({"type": "error", "detail": str(exc)})

        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/generate/stream")
    async def generate_stream(req: StreamGenerateRequest):
        """Stream generated tokens AND each token's J-space, live via SSE.

        For every token the model produces we emit its text and the per-layer
        top concepts of the residual that produced it. Stops on EOS.
        """
        import threading

        async def gen():
            engine = service.engine
            loop = asyncio.get_running_loop()
            # Build the starting context.
            if req.messages:
                msgs = [{"role": m.get("role"), "content": m.get("content")}
                        for m in req.messages]
                input_ids = engine.model.build_chat_ids(
                    msgs, add_generation_prompt=True)
            else:
                input_ids = engine.model.build_input_ids(
                    req.prompt, req.use_chat_template)
            num_layers = engine.model.info.num_layers
            layers = req.layers or list(range(num_layers + 1))
            layers = sorted({l for l in layers if 0 <= l <= num_layers})

            if req.method == "jacobian":
                yield _sse({"type": "status", "detail": "fitting lens…"})
                fit_task = loop.run_in_executor(
                    None, lambda: engine.ensure_fitted(layers, None))
                # Emit heartbeats while the (possibly minutes-long) fit runs so
                # the connection isn't closed by the browser/proxy as idle.
                while not fit_task.done():
                    try:
                        await asyncio.wait_for(asyncio.shield(fit_task), timeout=5)
                    except asyncio.TimeoutError:
                        yield ": keep-alive\n\n"
                await fit_task  # surface any exception
            yield _sse({"type": "start", "layers": layers, "method": req.method})

            q: asyncio.Queue = asyncio.Queue()
            SENTINEL = object()

            def worker():
                try:
                    for ev in engine.stream_generate_with_lens(
                        input_ids, layers=layers, method=req.method,
                        max_new_tokens=req.max_new_tokens, top_k=req.top_k,
                        do_sample=req.temperature > 0, temperature=req.temperature,
                        top_p=req.top_p, sampling_top_k=req.sampling_top_k,
                        min_p=req.min_p,
                    ):
                        loop.call_soon_threadsafe(q.put_nowait, ev)
                except Exception as exc:
                    logger.exception("generate stream failed")
                    loop.call_soon_threadsafe(
                        q.put_nowait, {"type": "error", "detail": str(exc)})
                finally:
                    loop.call_soon_threadsafe(q.put_nowait, SENTINEL)

            threading.Thread(target=worker, daemon=True).start()
            while True:
                ev = await q.get()
                if ev is SENTINEL:
                    break
                yield _sse(ev)
            yield _sse({"type": "done"})

        return StreamingResponse(gen(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.post("/api/decompose")
    def decompose(req: DecomposeRequest):
        try:
            input_ids = None
            if req.messages:
                msgs = [{"role": m.get("role"), "content": m.get("content")}
                        for m in req.messages]
                input_ids = service.engine.model.build_chat_ids(
                    msgs, add_generation_prompt=req.at_generation_point
                )
            dec = service.decomposer.decompose(
                req.prompt, layer=req.layer, position=req.position,
                corpus=req.corpus, use_chat_template=req.use_chat_template,
                input_ids=input_ids,
            )
            return dec.to_dict()
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("decompose failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/translate")
    def translate(req: TranslateRequest):
        """Gloss non-English tokens to English (batched + cached)."""
        try:
            mapping = service.translator.translate_batch(req.tokens)
            return {"translations": mapping}
        except Exception as exc:
            logger.exception("translate failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/tokenize")
    def tokenize(req: TokenizeRequest):
        """Map a word/phrase to a single representative vocabulary token id.

        For a swap target we want one J-lens token direction, so we take the
        first token of the (space-prefixed) word -- the form the J-lens usually
        surfaces for a concept.
        """
        try:
            tok = service.engine.model.tokenizer
            text = req.text.strip()
            # Prefer the leading-space form (" rugby"), which matches how the
            # J-lens indexes mid-sentence concept tokens.
            candidates = [" " + text, text]
            token_id = None
            for c in candidates:
                ids = tok.encode(c, add_special_tokens=False)
                if ids:
                    token_id = ids[0]
                    break
            if token_id is None:
                raise ValueError(f"could not tokenize {req.text!r}")
            return {"token_id": int(token_id),
                    "token": tok.decode([token_id])}
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        except Exception as exc:
            logger.exception("tokenize failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/chat")
    def chat(req: ChatRequest):
        """Generate the assistant's reply to a conversation."""
        try:
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            reply = service.engine.model.chat(
                messages, max_new_tokens=req.max_new_tokens,
                temperature=req.temperature, top_p=req.top_p,
                top_k=req.top_k, min_p=req.min_p,
            )
            return {"reply": reply}
        except Exception as exc:
            logger.exception("chat failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/chat/lens")
    def chat_lens(req: ChatLensRequest):
        """Run the lens over a conversation.

        With ``at_generation_point`` the assistant-turn header is appended, so
        the last position's readout is exactly what the model is poised to say
        as it begins its reply -- a window into its unspoken reasoning.
        """
        try:
            engine = service.engine
            messages = [{"role": m.role, "content": m.content} for m in req.messages]
            input_ids = engine.model.build_chat_ids(
                messages, add_generation_prompt=req.at_generation_point
            )
            if req.method == "jacobian":
                corpus = req.corpus or None
                result = engine.jacobian_lens(
                    prompt="", layers=req.layers, positions=req.positions,
                    corpus=corpus, top_k=req.top_k, input_ids=input_ids,
                )
            else:
                result = engine.logit_lens(
                    prompt="", layers=req.layers, positions=req.positions,
                    top_k=req.top_k, input_ids=input_ids,
                )
            return result.to_dict()
        except Exception as exc:
            logger.exception("chat lens failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/intervene")
    def intervene(req: InterveneRequest):
        """Generate with J-space edits (steer / ablate / swap), plus an
        optional unedited baseline for comparison."""
        try:
            engine = service.engine
            if req.messages:
                messages = [{"role": m.role, "content": m.content}
                            for m in req.messages]
                input_ids = engine.model.build_chat_ids(
                    messages, add_generation_prompt=True
                )
            else:
                input_ids = engine.model.build_input_ids(
                    req.prompt or "", req.use_chat_template
                )
            edits = [iv.model_dump() for iv in req.interventions]
            from .model import build_sampling_kwargs
            do_sample = req.temperature > 0
            gen_kwargs = {"do_sample": do_sample}
            gen_kwargs.update(build_sampling_kwargs(
                do_sample, req.temperature, top_p=req.top_p,
                top_k=req.top_k, min_p=req.min_p,
            ))
            steered = engine.generate_with_interventions(
                input_ids, edits, max_new_tokens=req.max_new_tokens, **gen_kwargs
            )
            result = {"steered": steered, "interventions": edits}
            if req.include_baseline:
                result["baseline"] = engine.generate_with_interventions(
                    input_ids, [], max_new_tokens=req.max_new_tokens, **gen_kwargs
                )
            return result
        except Exception as exc:
            logger.exception("intervene failed")
            raise HTTPException(status_code=500, detail=str(exc))

    @app.post("/api/generate")
    def generate(req: GenerateRequest):
        try:
            text = service.engine.model.generate(
                req.prompt, max_new_tokens=req.max_new_tokens,
                use_chat_template=req.use_chat_template,
            )
            return {"text": text}
        except Exception as exc:
            logger.exception("generate failed")
            raise HTTPException(status_code=500, detail=str(exc))

    # Static frontend.
    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/")
        def index():
            return FileResponse(str(STATIC_DIR / "index.html"))

    @app.get("/favicon.ico")
    def favicon():
        from fastapi.responses import Response
        return Response(status_code=204)

    return app


def _sse(obj: dict) -> str:
    return f"data: {json.dumps(obj)}\n\n"


# Headers that keep Server-Sent Events flowing through proxies / browsers:
# no caching, no buffering (nginx honours X-Accel-Buffering), keep-alive.
SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "X-Accel-Buffering": "no",
    "Connection": "keep-alive",
}


app = create_app()
