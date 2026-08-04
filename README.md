# 🌴 AI Quality Control Copilot

Built for **JPG Hackathon 2026: AI for Oil Quality Challenge**.

An MVP pipeline that predicts Crude Palm Oil (CPO) Free Fatty Acid (FFA)
degradation from fruit ripeness + intake/storage conditions, and turns the
prediction into a plain-language operational action plan.

## Architecture

| Module | File(s) | What it does |
|---|---|---|
| 1 · Vision Grader | `vision_grader.py` | Mocked CV: color-informed weighted-random ripeness classification (Underripe/Ripe/Overripe/Rotted) from an uploaded FFB photo. A clearly-labeled, swappable placeholder for a future trained YOLO detector. |
| 2 · Predictive Engine | `generate_data.py`, `ml_engine.py` | Synthetic dataset generated from a domain-informed FFA formula; XGBoost Regressor trained and cached (`st.cache_resource`) for real-time inference. |
| 3 · AI Copilot | `copilot.py` | Packages Module 1 + 2 outputs into a structured JSON payload, sends it to an LLM (Claude or GPT) prompted as a Palm Oil Mill Operations Expert, and returns a concise action plan. Falls back to a rule-based recommendation if no API key is set or the live call fails. |
| App shell | `app.py`, `ui_components.py` | Streamlit layout, sidebar controls, and the custom card/banner/chat/gauge UI. |

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

That's it — on first launch, the app automatically generates
`synthetic_batch_data.csv` and a handful of demo sample images if they don't
already exist, then trains and caches the XGBoost model. No manual setup
scripts required, though you can run `python generate_data.py` on its own if
you just want to inspect the dataset.

**Python 3.9+** required.

## Using the AI Copilot

Open the sidebar and choose a provider:

- **Anthropic Claude** — paste an API key from [console.anthropic.com](https://console.anthropic.com). Default model: `claude-sonnet-5`.
- **OpenAI GPT** — paste an API key from [platform.openai.com](https://platform.openai.com). Default model: `gpt-5.4-mini`.

Both model names are editable under "Advanced: model name" in the sidebar,
since provider model lineups move fast — if a default ever 404s, drop in
whatever current model ID your account has access to.

**No key? No problem.** Leave the field blank (or if a live call fails for
any reason — bad key, rate limit, flaky conference wifi) and the Copilot
automatically uses a rule-based fallback recommendation engine instead of
breaking the demo. A small caption always tells you which mode produced the
answer on screen.

## Design notes

- **Color-coding**: Green `<2.5%`, Amber `2.5–3.5%`, Red `>3.5%` FFA — used consistently across the status strip, gauge, banners, and badges.
- **Custom styling**: cards/banners/chat are hand-written CSS using Tailwind's own color and spacing scale, injected once via a single `<style>` block. (The Tailwind CDN `<script>` doesn't reliably execute when injected through `st.markdown`, since browsers don't run `<script>` tags inserted via innerHTML — this sidesteps that issue while keeping the same visual system.)
- **Session stability**: the mock vision result is cached per-image-hash in `st.session_state`, so it won't re-roll every time you move an unrelated slider — only when a genuinely new photo/sample is analyzed.

## Project structure

```
.
├── app.py                     # Streamlit app: layout, sidebar, tabs
├── vision_grader.py            # Module 1: mocked CV ripeness grader
├── generate_data.py            # Synthetic dataset generator
├── ml_engine.py                 # Module 2: XGBoost training + inference
├── copilot.py                  # Module 3: LLM integration + offline fallback
├── ui_components.py            # Custom CSS + HTML component builders, Plotly charts
├── requirements.txt
└── README.md
```
