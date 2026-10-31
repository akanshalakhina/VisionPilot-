import { useCallback, useEffect, useState } from "react";
import { api } from "./api.js";
import { money } from "./format.js";
import AskForm from "./components/AskForm.jsx";
import AuditLog from "./components/AuditLog.jsx";
import Learned from "./components/Learned.jsx";
import RunCard from "./components/RunCard.jsx";

export default function App() {
  const [state, setState] = useState(null);
  const [runs, setRuns] = useState([]);
  const [audit, setAudit] = useState([]);
  const [error, setError] = useState("");
  const [loaded, setLoaded] = useState(false);
  const [busy, setBusy] = useState(null); // what is in flight, e.g. "plan" or "execute:3"

  const refresh = useCallback(async () => {
    try {
      const [nextState, nextRuns, nextAudit] = await Promise.all([api.state(), api.runs(), api.audit()]);
      setState(nextState);
      setRuns(nextRuns);
      setAudit(nextAudit);
    } catch (problem) {
      setError(`Cannot reach the backend: ${problem.message}`);
    } finally {
      setLoaded(true);
    }
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  // Run one API call, show the server's message if it refuses, and reload everything either way.
  const act = useCallback(
    async (key, call) => {
      setBusy(key);
      setError("");
      try {
        await call();
      } catch (problem) {
        setError(problem.message);
      } finally {
        await refresh();
        setBusy(null);
      }
    },
    [refresh],
  );

  const names = state ? state.variants.map((variant) => variant.name) : [];
  const proposed = runs.filter((run) => run.status === "proposed").length;

  return (
    <div className="page">
      <header className="top">
        <div>
          <h1>VisionPilot</h1>
          <p className="muted">A learning agent for small-business ad budgets. It proposes, you decide.</p>
        </div>
        {state && (
          <ul className="badges" aria-label="System status">
            <li className="badge warn">Simulated data</li>
            <li className="badge">{state.planner === "llm" ? "LLM explains" : "Offline planner"}</li>
            <li className="badge">Cap {money(state.max_total_budget)}</li>
            <li className="badge">Minimum share {Math.round(state.exploration_floor * 100)}%</li>
          </ul>
        )}
      </header>

      <div role="alert" aria-live="polite">
        {error && (
          <p className="banner">
            {error}
            <button type="button" className="link" onClick={() => setError("")}>
              Dismiss
            </button>
          </p>
        )}
      </div>

      {!loaded && <p className="muted">Loading…</p>}

      {state && (
        <main className="grid">
          <section className="col">
            <AskForm cap={state.max_total_budget} busy={busy === "plan"} onSubmit={(goal, budget) => act("plan", () => api.plan(goal, budget))} />

            <div className="section-head">
              <h2>Runs</h2>
              <span className="muted">{proposed ? `${proposed} waiting for your decision` : "newest first"}</span>
            </div>
            {runs.length === 0 && <p className="empty">No runs yet. Ask the agent for a plan above.</p>}
            {runs.map((run) => (
              <RunCard
                key={run.id}
                run={run}
                names={names}
                busy={busy}
                onApprove={(approved) => act(`approve:${run.id}`, () => api.approve(run.id, approved))}
                onExecute={() => act(`execute:${run.id}`, () => api.execute(run.id))}
              />
            ))}
          </section>

          <aside className="col">
            <Learned state={state} />
            <AuditLog events={audit} />
          </aside>
        </main>
      )}
    </div>
  );
}
