# The agent

The agent turns an owner's goal and budget into a **proposed** plan, explains it in plain language, and waits for approval. It never runs anything itself.

```
POST /agent/plan  ->  proposed  ->  POST .../approve  ->  approved  ->  POST .../execute  ->  executed (sandbox)
                                       (or rejected)                         |
                                                                             +-> results update the agent's memory
```

## Tools

| Tool | What it does | Backed by |
|---|---|---|
| `list_ad_variants` | The ads that can be tested | catalog |
| `estimate_variant_ctr` | Click rate per ad with a 90% credible interval. `null` means no results yet | Beta belief learned from executed runs |
| `recall_past_runs` | Summaries of the last executed plans | memory |
| `recommend_budget_split` | Budget shares per ad | Thompson sampling with a 1% minimum per ad (`EXPLORATION_FLOOR`) |
| `rank_audience` | Expected extra visits from targeting the top share of customers | X-learner uplift model |
| `propose_plan` | Creates the proposal. Call once | bandit split, guardrails |

There is **no tool that spends money or runs a plan.**

## Safety properties (each one has a test)

- A plan's budget must equal the owner's budget, and must not exceed `MAX_TOTAL_BUDGET`. The LLM cannot change either.
- The budget split in a plan always comes from the bandit. The LLM cannot supply its own split.
- Execution needs the owner's approval, and the guardrails are checked again at execution time.
- **Grounding check:** every number in the explanation must appear in a tool result or in the owner's request, to the precision written. A bad LLM answer gets one rewrite. If it still contains an unbacked number, the explanation is built from a template instead, and the rejected text is stored in the run record.
- If the LLM fails or is not configured, the offline rule-based planner produces the plan.
- Every proposal, approval, rejection and execution is written to the audit log.

## Running it

Offline (default, no key needed):

```powershell
uvicorn backend.main:create_app --factory
```

```powershell
$body = @{ goal = "Get more customers for my festive sale"; budget = 1000 } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/agent/plan -ContentType "application/json" -Body $body
```

Then `POST /agent/runs/{id}/approve` with `{"approved": true}` and `POST /agent/runs/{id}/execute`. Interactive docs are at `http://127.0.0.1:8000/docs`.

With an LLM, set these in `.env` using your **own** account and a model from your provider's current list:

```
LLM_PROVIDER=openrouter      # or groq, gemini, ollama, or set LLM_BASE_URL
LLM_MODEL=<model name>
LLM_API_KEY=<your own key>
```

`GET /health` shows which planner is active. The key never appears in logs, reprs or run records.

## What is real and what is simulated

- Click results come from the synthetic ad simulator. Its click rates (roughly 15% to 30%) are far higher than real ads, so use it to show the learning loop, never as evidence about real campaigns.
- The audience figures come from a synthetic customer base. The model's own prediction tends to be more optimistic than the observed figure on held-out customers. The explanation shows both for that reason.
- The audience recommendation is **advisory**. The sandbox execution only simulates splitting the budget across ads, so targeting is not applied in execution yet. The two simulators are separate.
- The Avazu CTR model is a standalone component and is not called by the agent. Its inputs (hashed site, device and placement features) do not map to a business owner's ad variants. The agent's per-ad click estimates come from its own learned belief instead.

## Known limitations

- The LLM path has been tested with a scripted model and a mock HTTP transport only. It has never been run against a live provider. Free models can be unreliable at tool calling, which is what the fallbacks are for.
- The API has no authentication. "Owner approval" is an API call, not an identity check. Add login before this is exposed beyond a local demo.
- The grounding check finds numbers that no tool produced. It does not check that a real number is attached to the right ad, and it cannot see numbers written as words.
- One business only (`default`). Memory is a single shared belief.

## Evaluation hooks

Each run record stores the planner used, the full tool trace, and the grounding result (`numbers_checked`, `ungrounded`, `explanation_source`, whether the first LLM text was rejected). These are the raw material for the 50-task benchmark: ungrounded-number rate, fallback rate, tool-choice accuracy, and the ablation of rule-based planner versus LLM planner.
