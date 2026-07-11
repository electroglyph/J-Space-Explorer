# Contributing to J-Space Explorer

Thanks for your interest! This project is a research/education tool for exploring
the **Jacobian lens** and **J-space** of local language models.

## Development setup

```bash
git clone https://github.com/Sweaterdog/J-Space-Explorer.git
cd J-Space-Explorer
./setup.sh                      # creates the 'jspace' conda env + installs
conda activate jspace
export JSPACE_MODEL_PATH=/path/to/a/local/safetensors/model
./run.sh
```

The backend is Python (FastAPI + PyTorch/transformers); the frontend is a single
dependency-free static page (`src/jspace/web/static/`).

## Project layout

| Path | What |
|------|------|
| `src/jspace/model.py`   | model loading, residual-stream access, unembed, chat/generate |
| `src/jspace/lens.py`    | logit lens, Jacobian-lens fit/apply, J-space interventions |
| `src/jspace/jspace.py`  | sparse J-space decomposition |
| `src/jspace/translate.py` | live token→English glossing |
| `src/jspace/server.py`  | FastAPI endpoints (lens, chat, intervene, translate, fit) |
| `src/jspace/cli.py`     | `jspace` CLI (`serve`, `lens`, `fit`, `info`) |
| `src/jspace/web/static/`| frontend: `index.html`, `app.js`, `tour.js`, `style.css` |

## Guidelines

- **Correctness first.** The lens math is subtle; if you touch `lens.py`, sanity
  check against the logit lens (e.g. "Paris" should emerge for "The capital of
  France is") and, ideally, validate the Jacobian estimator numerically.
- **Keep the frontend dependency-free** (vanilla JS). No build step.
- **Match the existing style.** Python is gofmt-like clean (4-space, type hints,
  docstrings); JS uses the existing helpers (`$`, `escapeHtml`, `readSSE`).
- **Never commit** model weights, the fitted-lens cache (`*.pt`), or large data.
- Run a quick smoke test before opening a PR: `jspace info`, then a lens run in
  the browser.

## Reporting issues

Please include: model architecture (`jspace info`), `transformers`/`torch`
versions, GPU/VRAM, and the exact prompt + settings that reproduce the problem.

## License

By contributing you agree that your contributions are licensed under the
project's **Apache License 2.0**.
