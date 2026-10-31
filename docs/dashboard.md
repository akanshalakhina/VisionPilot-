# Dashboard

A single-page React app (Vite) that sits on top of the agent's REST API. It adds no new decision logic: it only shows what the backend returns and calls the same propose, approve and execute endpoints.

## Run it

Production build, served by the backend at `/`:

```powershell
cd frontend
npm install
npm run build
cd ..
uvicorn backend.main:create_app --factory --reload
```

Then open http://127.0.0.1:8000/. The backend mounts `frontend/dist` only if it exists when the app starts, so restart the server after the first build.

Development with hot reload, API calls forwarded to the backend on port 8000:

```powershell
uvicorn backend.main:create_app --factory --reload   # one terminal
cd frontend; npm run dev                              # another terminal, open http://localhost:5173
```

## What it shows

| Area | Source endpoint | Notes |
|---|---|---|
| Ask the agent for a plan | `POST /agent/plan` | The form checks the budget against the cap before sending. The server checks again and its message is shown if it refuses. |
| Runs | `GET /agent/runs` | Newest first. Each run shows the explanation, the budget split, and whether the grounding check passed. |
| Approve / Reject / Run | `POST /agent/runs/{id}/approve`, `/execute` | A button appears only in the state where it is allowed. A stale button gets the server's refusal as a banner, and the card then refreshes. |
| Tool trace | `GET /agent/runs/{id}` | Loaded when opened, because the trace is large and the list endpoint leaves it out. |
| What the agent has learned | `GET /agent/state` | Estimated click rate per ad with its 90% credible interval, and the executed runs. Empty until a plan has been run. |
| Audit log | `GET /audit` | The latest events: proposals, approvals, executions, refusals. |

Everything is labelled as simulated data, because the only adapter that runs is the sandbox.

## Limits

- No login. Anyone who can reach the server can approve and run plans. Do not expose it beyond your own machine.
- No automated frontend tests. It was checked by hand in a browser: propose, approve, run, reject, a second plan after learning, a stale button against the server, and the layout at phone width.
- The page loads its data once and after each action. It does not poll, so a change made elsewhere shows after the next action or a reload.
- Charts are plain SVG, written without a charting library.
