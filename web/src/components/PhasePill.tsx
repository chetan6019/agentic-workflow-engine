// Colored status pill for a run phase label (labels arrive from the API with emoji).
// Color rules ported from streamlit_app/app.py::_phase_color.

function phaseColor(label: string): string {
  const s = label.toLowerCase();
  if (s.includes("fail") || s.includes("❌")) return "red";
  if (s.includes("approval") || s.includes("🛑")) return "amber";
  if (s.includes("done") || s.includes("finaliz") || s.includes("💾")) return "green";
  return "purple";
}

export function PhasePill({ label, pulse = false }: { label: string; pulse?: boolean }) {
  return (
    <span className={`pill pill-${phaseColor(label)} ${pulse ? "pill-pulse" : ""}`}>
      {pulse && <span className="dot" />}
      {label}
    </span>
  );
}
