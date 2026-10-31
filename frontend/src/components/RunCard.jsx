import { useState } from "react";
import { api } from "../api.js";
import { money, percent, whole, when } from "../format.js";

function Allocation({ plan }) {
  const rows = Object.entries(plan.allocation);
  const top = Math.max(...rows.map(([, share]) => share));
  return (
    <ul className="alloc" aria-label="Budget split">
      {rows.map(([name, share]) => (
        <li key={name}>
          <span className="alloc-name">{name}</span>
          <span className="alloc-track" aria-hidden="true">
            <span className="alloc-fill" style={{ width: `${(share / top) * 100}%` }} />
          </span>
          <span className="alloc-num">
            {percent(share)} · {money(plan.amounts[name])}
          </span>
        </li>
      ))}
    </ul>
  );
}

function Grounding({ grounding }) {
  const clean = grounding.ok;
  const source = grounding.explanation_source === "llm" ? "language-model text" : "template text";
  return (
    <p className={`chip ${clean ? "good" : "bad"}`}>
      {clean
        ? `Grounded: all ${grounding.numbers_checked} numbers match tool results (${source})`
        : `Not grounded: ${grounding.ungrounded.length} number(s) with no source`}
    </p>
  );
}

function Result({ result, names }) {
  const ids = Object.keys(result.impressions);
  const totalClicks = ids.reduce((sum, id) => sum + result.clicks[id], 0);
  const totalImpressions = ids.reduce((sum, id) => sum + result.impressions[id], 0);
  return (
    <div className="result">
      <h4>Result (simulated)</h4>
      <table>
        <thead>
          <tr>
            <th>Ad</th>
            <th>Impressions</th>
            <th>Clicks</th>
            <th>Click rate</th>
          </tr>
        </thead>
        <tbody>
          {ids.map((id) => (
            <tr key={id}>
              <td>{names[id] ?? `Ad ${Number(id) + 1}`}</td>
              <td>{whole(result.impressions[id])}</td>
              <td>{whole(result.clicks[id])}</td>
              <td>{result.impressions[id] ? percent(result.clicks[id] / result.impressions[id]) : "–"}</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>Total · spend {money(result.spend)}</td>
            <td>{whole(totalImpressions)}</td>
            <td>{whole(totalClicks)}</td>
            <td>{totalImpressions ? percent(totalClicks / totalImpressions) : "–"}</td>
          </tr>
        </tfoot>
      </table>
    </div>
  );
}

function Trace({ id }) {
  const [trace, setTrace] = useState(null);
  const [problem, setProblem] = useState("");

  async function load(event) {
    if (event.target.open && trace === null) {
      try {
        setTrace((await api.run(id)).trace);
      } catch (error) {
        setProblem(error.message);
      }
    }
  }

  return (
    <details onToggle={load} className="trace">
      <summary>Tool trace (what the agent looked at)</summary>
      {problem && <p className="hint bad">{problem}</p>}
      {trace === null && !problem && <p className="muted">Loading…</p>}
      {trace && (
        <ol>
          {trace.map((step) => (
            <li key={step.id}>
              <strong>{step.tool}</strong> {step.ok ? "" : "(failed) "}
              <pre>{JSON.stringify(step.result, null, 1).slice(0, 700)}</pre>
            </li>
          ))}
        </ol>
      )}
    </details>
  );
}

export default function RunCard({ run, names, busy, onApprove, onExecute }) {
  const working = busy !== null && String(busy).endsWith(`:${run.id}`);
  const anyBusy = busy !== null;

  return (
    <article className="card run">
      <header className="run-head">
        <div>
          <h3>
            Run #{run.id} <span className={`pill ${run.status}`}>{run.status}</span>
          </h3>
          <p className="muted">
            {run.goal} · {money(run.budget)} · {when(run.ts)}
          </p>
        </div>
      </header>

      <p className="explain">{run.explanation}</p>
      <Allocation plan={run.plan} />
      <Grounding grounding={run.grounding} />

      {run.status === "proposed" && (
        <div className="row">
          <button type="button" className="primary" disabled={anyBusy} onClick={() => onApprove(true)}>
            {working ? "Working…" : "Approve"}
          </button>
          <button type="button" disabled={anyBusy} onClick={() => onApprove(false)}>
            Reject
          </button>
        </div>
      )}
      {run.status === "approved" && (
        <div className="row">
          <button type="button" className="primary" disabled={anyBusy} onClick={onExecute}>
            {working ? "Running…" : "Run in the simulator"}
          </button>
          <span className="muted">Spends only simulated money.</span>
        </div>
      )}
      {run.status === "rejected" && <p className="muted">Rejected. Nothing ran and it cannot be executed.</p>}
      {run.result && <Result result={run.result} names={names} />}

      <Trace id={run.id} />
    </article>
  );
}
