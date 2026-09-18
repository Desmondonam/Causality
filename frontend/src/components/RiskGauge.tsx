const CX = 120;
const CY = 120;
const R = 96;

function polarToCartesian(percent: number) {
  // 0% -> 180deg (left), 100% -> 0deg (right), sweeping over the top.
  const angleDeg = 180 - (percent / 100) * 180;
  const angleRad = (angleDeg * Math.PI) / 180;
  return {
    x: CX + R * Math.cos(angleRad),
    y: CY - R * Math.sin(angleRad),
  };
}

function arcPath(fromPercent: number, toPercent: number) {
  const start = polarToCartesian(fromPercent);
  const end = polarToCartesian(toPercent);
  const largeArc = toPercent - fromPercent > 50 ? 1 : 0;
  return `M ${start.x} ${start.y} A ${R} ${R} 0 ${largeArc} 1 ${end.x} ${end.y}`;
}

export function RiskGauge({ probabilityMalignant }: { probabilityMalignant: number }) {
  const pct = Math.round(probabilityMalignant * 1000) / 10;
  const needle = polarToCartesian(pct);

  const color = pct < 30 ? "var(--benign)" : pct < 70 ? "var(--warning)" : "var(--malignant)";

  return (
    <div className="flex flex-col items-center">
      <svg viewBox="0 0 240 150" className="w-full max-w-xs">
        <path d={arcPath(0, 30)} stroke="var(--benign)" strokeWidth="18" fill="none" strokeLinecap="round" opacity={0.85} />
        <path d={arcPath(30, 70)} stroke="var(--warning)" strokeWidth="18" fill="none" opacity={0.85} />
        <path d={arcPath(70, 100)} stroke="var(--malignant)" strokeWidth="18" fill="none" strokeLinecap="round" opacity={0.85} />
        <line x1={CX} y1={CY} x2={needle.x} y2={needle.y} stroke="var(--foreground)" strokeWidth="3" strokeLinecap="round" />
        <circle cx={CX} cy={CY} r="6" fill="var(--foreground)" />
        <text x={CX} y={CY + 34} textAnchor="middle" fontSize="28" fontWeight="700" fill={color}>
          {pct.toFixed(1)}%
        </text>
        <text x={CX} y={CY + 52} textAnchor="middle" fontSize="11" fill="var(--muted)">
          malignancy risk
        </text>
      </svg>
    </div>
  );
}
