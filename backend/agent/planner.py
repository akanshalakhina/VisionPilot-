"""The agent loop.

Two planners share one set of tools:
  rule-based  a fixed script that always calls the same tools in the same order and writes the
              explanation from a template. It needs no LLM, so it works offline and is the
              no-LLM baseline for the evaluation.
  LLM         the model chooses which tools to call and writes the explanation. Its text is
              checked by the grounding check. If a number is not backed by a tool result, it
              gets one chance to rewrite. If that fails, the template is used instead.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from backend.agent.grounding import check_grounding, claims_action_taken
from backend.agent.llm import LLMClient
from backend.agent.tools import DEFAULT_TARGET_FRACTION, AgentServices, ToolBox

SYSTEM_PROMPT = (
    "You are VisionPilot, a marketing planning agent for a small business. You only PROPOSE plans. "
    "You cannot spend money or run anything, and the owner must approve a plan before it runs.\n"
    "Work like this: use list_ad_variants, estimate_variant_ctr, recall_past_runs, recommend_budget_split "
    "and rank_audience as needed, then call propose_plan exactly once with the budget the owner gave. "
    "The budget split always comes from the tools and cannot be changed.\n"
    "Then write a short plain-language explanation for the owner. Every number you state must be copied "
    "from a tool result (rounding is fine) or come from the owner's request. Never invent figures. "
    "If a tool says there is no data yet, say so. All results are simulated."
)


@dataclass
class AgentOutcome:
    plan: dict | None
    explanation: str
    grounding: dict
    trace: list[dict]
    planner: str
    extras: dict = field(default_factory=dict)


def _last_result(trace: list[dict], tool: str) -> dict | None:
    for step in reversed(trace):
        if step["tool"] == tool and step["ok"]:
            return step["result"]
    return None


def render_explanation(trace: list[dict]) -> str:
    """Plain-text explanation built only from tool results, so every number is grounded by construction."""
    plan = _last_result(trace, "propose_plan")["plan"]
    split = _last_result(trace, "recommend_budget_split")
    estimates = _last_result(trace, "estimate_variant_ctr")
    audience = _last_result(trace, "rank_audience")
    past = _last_result(trace, "recall_past_runs")

    lines = [f"Plan '{plan['name']}' with a budget of ₹{plan['budget']:,.2f} (simulated)."]
    if split:
        parts = [f"{r['name']} {r['share'] * 100:.1f}% (₹{r['amount']:,.2f})" for r in split["split"]]
        lines.append(
            f"Budget split by Thompson sampling, with a minimum of {split['exploration_floor'] * 100:.0f}% "
            f"for every ad so none is dropped: " + ", ".join(parts) + "."
        )
    if estimates:
        seen = [r for r in estimates["variants"] if r["impressions_seen"] > 0]
        if not seen:
            lines.append("No results have been observed yet, so this first split is exploratory.")
        else:
            level = estimates["credible_interval_percent"]
            for r in seen:
                lines.append(
                    f"{r['name']}: estimated click rate {r['estimated_ctr'] * 100:.1f}% "
                    f"({level}% interval {r['ctr_low'] * 100:.1f}% to {r['ctr_high'] * 100:.1f}%) "
                    f"from {r['impressions_seen']:,} impressions."
                )
    if audience:
        sentence = (
            f"Audience: targeting the top {audience['target_fraction'] * 100:.1f}% of customers by predicted uplift is "
            f"expected to add about {audience['predicted_extra_visits_targeted']:.1f} visits per "
            f"{audience['customers_basis']:,} customers targeted, against "
            f"{audience['predicted_extra_visits_everyone']:.1f} across everyone (model estimates)."
        )
        observed = audience.get("observed_extra_visits_targeted")
        if observed is not None:
            sentence += f" On held-out customers the observed figure was {observed:.1f}, which is noisy."
        lines.append(sentence)
    if past and past.get("runs"):
        last = past["runs"][-1]
        lines.append(
            f"The most recent executed run spent ₹{last['spend']:,.2f} and got {last['total_clicks']:,} clicks; "
            f"its best ad was {last['best_variant']}."
        )
    lines.append("Nothing runs until you approve this plan.")
    return "\n".join(lines)


class AgentRunner:
    def __init__(self, services: AgentServices, llm: LLMClient | None = None, max_steps: int = 6) -> None:
        self.services = services
        self.llm = llm
        self.max_steps = max_steps

    # ---- public ----------------------------------------------------------

    def run(self, goal: str, budget: float) -> AgentOutcome:
        extras: dict = {}
        toolbox = ToolBox(self.services, budget)
        text: str | None = None
        messages: list[dict] = []
        planner = "rule_based"

        if self.llm is not None:
            try:
                text, messages = self._llm_phase(toolbox, goal, budget)
                planner = "llm"
            except Exception as exc:  # network errors, bad responses: fall back to the offline planner
                extras["llm_error"] = f"{type(exc).__name__}: {exc}"
                toolbox = ToolBox(self.services, budget)
                planner = "rule_based_after_llm_error"

        if planner == "llm" and toolbox.plan is None:
            toolbox.call("propose_plan", {"total_budget": budget, "target_fraction": DEFAULT_TARGET_FRACTION})
            planner = "llm_without_proposal"
        if planner.startswith("rule_based"):
            self._script(toolbox, budget)

        if toolbox.plan is None:  # the budget was rejected by the guardrails
            error = next((s["result"].get("error") for s in reversed(toolbox.trace) if not s["ok"]), "no plan")
            return AgentOutcome(None, f"No plan was created: {error}", {"ok": False, "numbers_checked": 0,
                                "ungrounded": [], "explanation_source": "none"}, toolbox.trace, planner, extras)

        explanation, grounding = self._finalise(toolbox, text, messages, goal, budget, extras)
        return AgentOutcome(toolbox.plan, explanation, grounding, toolbox.trace, planner, extras)

    # ---- internals -------------------------------------------------------

    def _script(self, toolbox: ToolBox, budget: float) -> None:
        for tool, args in (
            ("list_ad_variants", {}),
            ("estimate_variant_ctr", {}),
            ("recall_past_runs", {}),
            ("recommend_budget_split", {"total_budget": budget}),
            ("rank_audience", {"target_fraction": DEFAULT_TARGET_FRACTION}),
            ("propose_plan", {"total_budget": budget, "target_fraction": DEFAULT_TARGET_FRACTION}),
        ):
            toolbox.call(tool, args)

    def _llm_phase(self, toolbox: ToolBox, goal: str, budget: float) -> tuple[str | None, list[dict]]:
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Owner's goal: {goal}\nBudget: {budget:g}"},
        ]
        schemas = toolbox.schemas()
        for _ in range(self.max_steps):
            reply = self.llm.chat(messages, tools=schemas)
            if not reply.tool_calls:
                return reply.content, messages
            messages.append({
                "role": "assistant",
                "content": reply.content,
                "tool_calls": [
                    {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                    for c in reply.tool_calls
                ],
            })
            for call in reply.tool_calls:
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(toolbox.call(call.name, call.arguments))})
        messages.append({"role": "user", "content": "Now write your final explanation for the owner."})
        return self.llm.chat(messages, tools=None).content, messages

    def _finalise(self, toolbox: ToolBox, text: str | None, messages: list[dict], goal: str,
                  budget: float, extras: dict) -> tuple[str, dict]:
        results = [s["result"] for s in toolbox.trace if s["ok"]]
        request = (goal, f"{budget:g}")

        def inspect(candidate: str):
            return check_grounding(candidate, results, request), claims_action_taken(candidate)

        if text and text.strip():
            report, claim = inspect(text)
            if (not report.ok or claim) and self.llm is not None:
                extras["first_llm_text_rejected"] = {
                    "text": text, "ungrounded": report.ungrounded, "claims_action": claim,
                }
                issues = []
                if report.ungrounded:
                    issues.append(f"These numbers do not appear in any tool result: {', '.join(report.ungrounded)}.")
                if claim:
                    issues.append("The text says an action was already taken, but nothing has been run and "
                                  "nothing runs until the owner approves.")
                retry = messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": " ".join(issues) + " Rewrite the explanation using only numbers "
                                                                   "from the tool results."},
                ]
                try:
                    text = self.llm.chat(retry, tools=None).content or ""
                    report, claim = inspect(text)
                except Exception as exc:
                    extras["llm_error"] = f"{type(exc).__name__}: {exc}"
                    report = None
            if report is not None and report.ok and not claim:
                # The text was checked only against what the LLM actually saw. Run any evidence
                # tools it skipped now, so the stored record always holds the split and estimates.
                self._script_missing(toolbox, budget)
                return text.strip(), {**report.as_dict(), "claims_action": False, "explanation_source": "llm"}
            extras["llm_text_rejected"] = True

        self._script_missing(toolbox, budget)  # the LLM may have skipped tools the template relies on
        results = [s["result"] for s in toolbox.trace if s["ok"]]
        explanation = render_explanation(toolbox.trace)
        report = check_grounding(explanation, results, request)
        return explanation, {**report.as_dict(), "claims_action": claims_action_taken(explanation),
                             "explanation_source": "template"}

    def _script_missing(self, toolbox: ToolBox, budget: float) -> None:
        wanted = (
            ("estimate_variant_ctr", {}),
            ("recall_past_runs", {}),
            ("recommend_budget_split", {"total_budget": budget}),
            ("rank_audience", {"target_fraction": toolbox.plan["target_fraction"]}),
        )
        have = {s["tool"] for s in toolbox.trace if s["ok"]}
        for tool, args in wanted:
            if tool not in have:
                toolbox.call(tool, args)
