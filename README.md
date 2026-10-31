# VisionPilot

A learning AI agent for small-business digital marketing.

The owner describes a goal and a budget. The agent predicts which ads and audiences will perform, splits the budget, keeps learning from results, writes the copy and explains each decision. The owner approves every action, and actions run in a **sandbox simulator**.

The decisions come from models trained in this repo (click prediction, a budget bandit, uplift targeting). An LLM plans, calls those models as tools and explains the results. It does not decide.

## Status

| Part | State |
|---|---|
| Synthetic ad environment, bandit policies (Thompson, LinUCB), off-policy evaluation (IPW, SNIPW, DM, DR) | Working, tested |
| FastAPI backend: sandbox adapter, guardrails (budget cap, owner approval), SQLite audit log | Working, tested |
| Meta adapter | Disabled placeholder. Not implemented |
| Data loaders for Avazu, Criteo Uplift, Open Bandit | Written. Tested on fabricated files only, not yet run on the real datasets |
| CTR predictor: logistic regression baseline vs LightGBM, split by day, calibration metrics | Written. Tested on fabricated data only. Real training runs on Kaggle, see `docs/kaggle_training.md` |
| Benchmark: 50 tasks, safety layer vs 6 misbehaving scripted LLMs, decision quality vs baselines in 6 synthetic settings | Written and run. All tasks pass. Thompson sampling did not beat simple greedy. The exploration floor was lowered from 5% to 1% after the benchmark showed it cost clicks, see `docs/benchmark.md`. The real-LLM parts (`--llm`) have not been run |
| Uplift targeting: S-, T- and X-learner vs a response baseline, Qini/AUUC, bootstrap intervals | Written. Works on the synthetic generator (known true effect). Not yet run on real Criteo data, see `docs/kaggle_training.md` |
| Agent shell: tool-using planner (offline rule-based or any OpenAI-compatible LLM), memory that learns from executed runs, grounding check on every number, propose / approve / execute flow | Written and tested. The LLM path is tested with a scripted model and a mock transport only, never against a live provider. See `docs/agent.md` |
| React dashboard: ask for a plan, approve or reject it, run it in the simulator, see what the agent has learned, read the audit log | Built and checked by hand in a browser, with no automated frontend tests. It has no login, so anyone who can reach the server can approve. See `docs/dashboard.md` |

## Quick start (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest
python -m ml.bandit.experiment --seed 0 --out reports/bandit_demo.json
cd frontend; npm install; npm run build; cd ..
uvicorn backend.main:create_app --factory --reload
```

Open http://127.0.0.1:8000/ for the dashboard and http://127.0.0.1:8000/docs for the API. Without `npm run build` the backend still works and serves only the API.

Copy `.env.example` to `.env` to change settings. Never commit `.env`.

## Layout

```
backend/    FastAPI app, execution adapters, guardrails, audit log, the agent
ml/         data loaders, simulator, bandit policies, off-policy evaluation, CTR predictor, uplift
benchmark/  the 50-task benchmark, misbehaving scripted LLMs, decision-quality study
frontend/   React dashboard (Vite). `npm run build` writes frontend/dist, which the backend serves at /
docs/       agent design, benchmark write-up, dashboard, how to train on Kaggle
tests/      pytest suite (fabricated fixtures in tests/fakes.py)
reports/    experiment outputs (small JSON)
```

## Honesty notes

- Everything from `ml/simulator` is **synthetic**. It tests the learning code. It is not evidence about real campaigns, so label it that way in any report or demo.
- The sandbox adapter simulates results. No real ads are run and no real money is spent.
- Datasets must be downloaded separately. Check each dataset's terms and cite it.
- Use only your own accounts and keys. Never use someone else's tokens or ad accounts.
