// Thin wrapper around the backend's REST API. Every call returns parsed JSON or throws an Error
// whose message is the server's own explanation (FastAPI puts it in `detail`).

function describe(detail) {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((item) => item.msg ?? JSON.stringify(item)).join("; ");
  return JSON.stringify(detail);
}

async function request(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = describe(body.detail ?? body);
    } catch {
      // the body was not JSON: keep the status line
    }
    throw new Error(message);
  }
  return response.json();
}

const post = (path, body) => request(path, { method: "POST", body: body === undefined ? undefined : JSON.stringify(body) });

export const api = {
  state: () => request("/agent/state"),
  runs: (limit = 30) => request(`/agent/runs?limit=${limit}`),
  run: (id) => request(`/agent/runs/${id}`),
  audit: (limit = 40) => request(`/audit?limit=${limit}`),
  plan: (goal, budget) => post("/agent/plan", { goal, budget }),
  approve: (id, approved) => post(`/agent/runs/${id}/approve`, { approved }),
  execute: (id) => post(`/agent/runs/${id}/execute`),
};
