import { summarise, when, words } from "../format.js";

export default function AuditLog({ events }) {
  return (
    <section className="card">
      <h2>Audit log</h2>
      {events.length === 0 ? (
        <p className="empty">No events yet.</p>
      ) : (
        <ul className="audit">
          {events.map((event) => (
            <li key={event.id}>
              <span className="audit-time">{when(event.ts)}</span>
              <span className="audit-action">
                <strong>{event.actor}</strong> · {words(event.action)}
              </span>
              <span className="audit-detail muted">{summarise(event.payload)}</span>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
