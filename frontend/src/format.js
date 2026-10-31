export const money = (value) =>
  `₹${Number(value).toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

export const percent = (fraction, digits = 1) => `${(fraction * 100).toFixed(digits)}%`;

export const whole = (value) => Number(value).toLocaleString("en-IN");

export function when(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString(undefined, { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

// "agent_plan_approved" -> "agent plan approved"
export const words = (snake) => String(snake).replaceAll("_", " ");

// A short, readable line for an audit event's payload.
export function summarise(payload) {
  if (!payload || typeof payload !== "object") return "";
  const parts = [];
  if (payload.run_id !== undefined) parts.push(`run #${payload.run_id}`);
  if (payload.budget !== undefined) parts.push(money(payload.budget));
  if (payload.planner) parts.push(`${words(payload.planner)} planner`);
  if (payload.grounding_ok !== undefined) parts.push(payload.grounding_ok ? "grounded" : "not grounded");
  if (payload.reason) parts.push(String(payload.reason));
  return parts.join(" · ") || JSON.stringify(payload).slice(0, 120);
}
