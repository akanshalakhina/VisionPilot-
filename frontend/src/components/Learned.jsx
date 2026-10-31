import { money, percent, whole } from "../format.js";

const W = 460;
const LEFT = 58;
const RIGHT = 12;
const ROW = 34;

// One row per ad: a line for the credible interval and a dot for the estimate, all on one shared axis.
function Chart({ variants, level }) {
  const top = Math.max(...variants.map((variant) => variant.ctr_high));
  const max = Math.ceil((top * 1.1) / 0.05) * 0.05;
  const x = (value) => LEFT + (value / max) * (W - LEFT - RIGHT);
  const ticks = Array.from({ length: Math.round(max / 0.05) + 1 }, (_, i) => i * 0.05).filter((_, i, all) => all.length <= 8 || i % 2 === 0);
  const height = 24 + variants.length * ROW;

  return (
    <svg viewBox={`0 0 ${W} ${height}`} role="img" aria-label={`Estimated click rate per ad with ${level}% credible intervals`} className="chart">
      {ticks.map((tick) => (
        <g key={tick}>
          <line x1={x(tick)} x2={x(tick)} y1={4} y2={height - 20} className="grid" />
          <text x={x(tick)} y={height - 6} textAnchor="middle" className="axis">
            {Math.round(tick * 100)}%
          </text>
        </g>
      ))}
      {variants.map((variant, i) => {
        const y = 18 + i * ROW;
        return (
          <g key={variant.name}>
            <text x={LEFT - 8} y={y + 4} textAnchor="end" className="label">
              {variant.name}
            </text>
            <line x1={x(variant.ctr_low)} x2={x(variant.ctr_high)} y1={y} y2={y} className="interval" />
            <circle cx={x(variant.estimated_ctr)} cy={y} r={5} className="dot" />
          </g>
        );
      })}
    </svg>
  );
}

export default function Learned({ state }) {
  const seen = state.variants.some((variant) => variant.impressions_seen > 0);

  return (
    <section className="card">
      <h2>What the agent has learned</h2>
      {seen ? (
        <>
          <Chart variants={state.variants} level={state.credible_interval_percent} />
          <p className="muted small">Dot: estimated click rate. Line: {state.credible_interval_percent}% credible interval. Simulated clicks.</p>
          <table className="compact">
            <thead>
              <tr>
                <th>Ad</th>
                <th>Click rate</th>
                <th>Seen</th>
              </tr>
            </thead>
            <tbody>
              {state.variants.map((variant) => (
                <tr key={variant.name}>
                  <td>{variant.name}</td>
                  <td>
                    {percent(variant.estimated_ctr)} ({percent(variant.ctr_low)}–{percent(variant.ctr_high)})
                  </td>
                  <td>
                    {whole(variant.clicks_seen)} / {whole(variant.impressions_seen)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      ) : (
        <p className="empty">Nothing observed yet. Approve and run a plan, and the agent will learn from the clicks.</p>
      )}

      {state.history.length > 0 && (
        <>
          <h3>Executed runs</h3>
          <table className="compact">
            <thead>
              <tr>
                <th>Run</th>
                <th>Spend</th>
                <th>Clicks</th>
                <th>Best ad</th>
              </tr>
            </thead>
            <tbody>
              {[...state.history].reverse().map((entry) => (
                <tr key={entry.run_id}>
                  <td>#{entry.run_id}</td>
                  <td>{money(entry.spend)}</td>
                  <td>{whole(entry.total_clicks)}</td>
                  <td>{entry.best_variant}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </>
      )}
    </section>
  );
}
