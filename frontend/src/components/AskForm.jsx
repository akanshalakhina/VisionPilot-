import { useState } from "react";
import { money } from "../format.js";

export default function AskForm({ cap, busy, onSubmit }) {
  const [goal, setGoal] = useState("Get more clicks for my shop");
  const [budget, setBudget] = useState("1000");

  const amount = Number(budget);
  const budgetOk = budget.trim() !== "" && Number.isFinite(amount) && amount > 0 && amount <= cap;
  const ready = goal.trim().length > 0 && budgetOk;

  function submit(event) {
    event.preventDefault();
    if (ready && !busy) onSubmit(goal.trim(), amount);
  }

  return (
    <form className="card ask" onSubmit={submit}>
      <h2>Ask the agent for a plan</h2>
      <label>
        Goal
        <input value={goal} maxLength={500} onChange={(event) => setGoal(event.target.value)} />
      </label>
      <label>
        Budget (₹)
        <input
          type="number"
          inputMode="decimal"
          min="1"
          max={cap}
          step="any"
          value={budget}
          onChange={(event) => setBudget(event.target.value)}
          aria-invalid={!budgetOk}
        />
      </label>
      {!budgetOk && budget.trim() !== "" && <p className="hint bad">Enter an amount above 0 and up to {money(cap)}.</p>}
      <div className="row">
        <button type="submit" className="primary" disabled={!ready || busy}>
          {busy ? "Planning…" : "Propose a plan"}
        </button>
        <span className="muted">Nothing runs until you approve the plan.</span>
      </div>
    </form>
  );
}
