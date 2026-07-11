"""Command-line entry point for the J-Space Explorer.

Usage:
    jspace serve                 # launch the web app
    jspace lens "prompt" ...     # print a lens readout to the terminal
    jspace info                  # load the model and print its architecture
"""
from __future__ import annotations

import argparse
import logging
import sys

from .config import AppConfig


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model", help="Path to the model directory (safetensors).")
    p.add_argument("--no-chat-template", action="store_true",
                   help="Do not wrap the prompt in the chat template.")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(prog="jspace", description="J-Space Explorer")
    sub = parser.add_subparsers(dest="command", required=True)

    p_serve = sub.add_parser("serve", help="Run the web app.")
    _add_common(p_serve)
    p_serve.add_argument("--host", default=None)
    p_serve.add_argument("--port", type=int, default=None)
    p_serve.add_argument("--eager", action="store_true",
                         help="Load the model at startup instead of on first request.")

    p_info = sub.add_parser("info", help="Load the model and print its architecture.")
    _add_common(p_info)

    p_lens = sub.add_parser("lens", help="Print a lens readout to the terminal.")
    _add_common(p_lens)
    p_lens.add_argument("prompt")
    p_lens.add_argument("--method", choices=["logit", "jacobian"], default="logit")
    p_lens.add_argument("--layers", help="Comma-separated layer indices.")
    p_lens.add_argument("--top-k", type=int, default=5)
    p_lens.add_argument("--last-token", action="store_true",
                        help="Only show the last position.")

    p_fit = sub.add_parser("fit", help="Fit and cache the Jacobian lens operators.")
    _add_common(p_fit)
    p_fit.add_argument("--layers", help="Comma-separated layer indices (default: all).")
    p_fit.add_argument("--corpus-size", type=int, default=None,
                       help="Number of corpus contexts to average over.")

    args = parser.parse_args(argv)
    config = AppConfig.from_env()
    if getattr(args, "model", None):
        config.model.path = args.model

    if args.command == "serve":
        return _serve(args, config)
    if args.command == "info":
        return _info(config)
    if args.command == "lens":
        return _lens(args, config)
    if args.command == "fit":
        return _fit(args, config)
    return 1


def _serve(args, config: AppConfig) -> int:
    import uvicorn
    from .server import create_app
    from .service import ModelService
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    service = ModelService(config)
    if args.eager:
        print("Eagerly loading model...")
        service.ensure_loaded()
    app = create_app(config, service=service)
    print(f"Serving on http://{config.host}:{config.port}")
    uvicorn.run(app, host=config.host, port=config.port)
    return 0


def _info(config: AppConfig) -> int:
    from .model import LensModel
    model = LensModel.load(config.model)
    i = model.info
    print(f"architecture : {i.architecture}")
    print(f"num_layers   : {i.num_layers}")
    print(f"hidden_size  : {i.hidden_size}")
    print(f"vocab_size   : {i.vocab_size}")
    print(f"dtype        : {i.dtype}")
    return 0


def _lens(args, config: AppConfig) -> int:
    from .lens import LensEngine
    from .model import LensModel
    model = LensModel.load(config.model)
    engine = LensEngine(model, config.lens)
    engine.config.top_k = args.top_k
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    use_chat = not args.no_chat_template

    if args.method == "jacobian":
        result = engine.jacobian_lens(args.prompt, layers=layers,
                                      use_chat_template=use_chat)
    else:
        result = engine.logit_lens(args.prompt, layers=layers,
                                   use_chat_template=use_chat)

    positions = list(range(len(result.tokens)))
    if args.last_token:
        positions = [len(result.tokens) - 1]

    # Group cells by layer for readable output.
    by_layer: dict[int, dict[int, list]] = {}
    for c in result.cells:
        by_layer.setdefault(c.layer, {})[c.position] = c.top

    print(f"\n=== {result.method} lens :: {args.prompt!r} ===")
    print("tokens:", " | ".join(result.tokens))
    for ell in sorted(by_layer):
        print(f"\n-- layer {ell} --")
        for pos in positions:
            top = by_layer[ell].get(pos)
            if not top:
                continue
            preview = ", ".join(f"{t.token!r}({t.score:.2f})" for t in top)
            print(f"  pos {pos:>3} {result.tokens[pos]!r:<14}: {preview}")
    return 0


def _fit(args, config: AppConfig) -> int:
    from .lens import LensEngine, default_corpus
    from .model import LensModel
    if args.corpus_size:
        config.lens.jacobian_corpus_size = args.corpus_size
    model = LensModel.load(config.model)
    engine = LensEngine(model, config.lens)
    layers = None
    if args.layers:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    else:
        layers = list(range(model.info.num_layers + 1))

    def progress(done, total):
        print(f"  fitting context {done}/{total}", flush=True)

    print(f"Fitting Jacobian lens for {len(layers)} layers over "
          f"{config.lens.jacobian_corpus_size} contexts...")
    # ensure_fitted fits missing layers (raw-text corpus, chat-template
    # independent), then persists to the disk cache.
    engine.ensure_fitted(layers, default_corpus(), progress=progress)
    if config.lens.jacobian_cache_path:
        print(f"Saved fitted lens to {config.lens.jacobian_cache_path}")
    print(f"Done. Fitted layers: {sorted(engine._jacobian.keys())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
