import { AnimatePresence, animate, motion, useMotionValue } from "motion/react";
import { useEffect, useLayoutEffect, useRef, useState } from "react";

type StageState = { status: string | null; attempts: number; running: boolean; reason: string | null };
type Decision = { agent: string; rule: string; outcome: unknown; rationale: string };
type RunView = {
  run_id: string;
  state: { stages: Record<string, StageState> };
  stages: string[];
  agent_of: Record<string, string>;
  models: Record<string, string>;
  decisions: Decision[];
  summaries: Record<string, { headline: string; facts: string[]; explain?: string; gates?: { name: string; outcome: string; meaning: string; detail: string }[] }>;
  evaluation: Evaluation | null;
};
type Cell = { mean: number; lift: number; verdict: string };
type Evaluation = {
  metrics: string[];
  reference: number[];
  reference_model: string;
  rows: { arm: string; fill: number; model: string; cells: Cell[] }[];
  findings: { arm: string; metric: string; verdict: string; text: string; n: number }[];
  sections: { title: string; paragraphs: string[] }[];
};

const EASE_OUT = [0.16, 1, 0.3, 1] as const;
const IN = 0.25; // entrance; exits are 20% shorter
const OUT = IN * 0.8;
const GLYPH: Record<string, string> = { SUCCESS: "✓", FAILED: "✕", RETRY: "↻", SKIPPED: "–", RUNNING: "●", PENDING: "○" };

const currentIndex = (run: RunView) => {
  const st = (s: string) => stageStatus(run.state.stages[s]);
  const running = run.stages.findIndex((s) => st(s) === "RUNNING");
  const open = run.stages.findIndex((s) => !["SUCCESS", "SKIPPED"].includes(st(s)));
  return running >= 0 ? running : open >= 0 ? open : run.stages.length - 1;
};
const label = (s: string) => s.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const stageStatus = (s?: StageState) => (s?.running ? "RUNNING" : (s?.status ?? "PENDING"));
const rowV = { hidden: { opacity: 0, y: 6 }, show: { opacity: 1, y: 0, transition: { duration: IN, ease: EASE_OUT } } };
// Every number on screen is shown to three decimals.
const r3 = (t: string) => t.replace(/(?<![\w.])-?\d+\.\d{4,}/g, (m) => Number(m).toFixed(3));
const short = (o: unknown) => {
  const t = r3(typeof o === "string" ? o : JSON.stringify(o));
  return t.length > 140 ? t.slice(0, 140) + "…" : t;
};
const METRIC = (m: string) => ({ f1: "F1", pr_auc: "PR-AUC", precision: "Precision", recall: "Recall" })[m] ?? m;
const tone = (v: string) => (v.includes("gain") ? "gain" : v.includes("loss") ? "loss" : "flat");
const sign = (n: number) => (n >= 0 ? "+" : "−") + Math.abs(n).toFixed(3);

function useJson<T>(url: string | null, every = 2000): T | null {
  const [data, setData] = useState<T | null>(null);
  useEffect(() => {
    if (!url) return;
    let alive = true;
    const load = () =>
      fetch(url).then((r) => (r.ok ? r.json() : null)).then((d) => alive && setData(d)).catch(() => {});
    load();
    const id = setInterval(load, every);
    return () => { alive = false; clearInterval(id); };
  }, [url, every]);
  return data;
}

function RunPicker({ runs, value, onChange }: { runs: string[]; value: string; onChange: (r: string) => void }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="picker">
      <motion.button className="btn" whileTap={{ scale: 0.97 }} onClick={() => setOpen(!open)}>
        {value}
      </motion.button>
      <AnimatePresence>
        {open && (
          <motion.div
            className="menu"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1, transition: { duration: 0.2, ease: EASE_OUT } }}
            exit={{ opacity: 0, scale: 0.95, transition: { duration: 0.16 } }}
          >
            {runs.map((r) => (
              <button key={r} onClick={() => { onChange(r); setOpen(false); }}>{r}</button>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// 'quoted' terms become orange text without the quotes; apostrophes inside words are left alone.
const quoted = (t: string) =>
  t.split(/(?<!\w)'([^']+)'(?!\w)/).map((part, i) => (i % 2 ? <span className="hl" key={i}>{part}</span> : part));

function ReportDropdown({ sections }: { sections: Evaluation["sections"] }) {
  const [open, setOpen] = useState(false);
  return (
    <div>
      <motion.button className="btn" style={{ marginTop: 10 }} whileTap={{ scale: 0.97 }} onClick={() => setOpen(!open)}>
        Full written report {open ? "−" : "+"}
      </motion.button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            style={{ overflow: "hidden" }}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1, transition: { duration: 0.25, ease: EASE_OUT } }}
            exit={{ height: 0, opacity: 0, transition: { duration: 0.2 } }}
          >
            {sections.map((sec) => (
              <div className="reportSec" key={sec.title}>
                <h4>{sec.title}</h4>
                {sec.paragraphs.map((p, i) => <p key={i}>{quoted(p)}</p>)}
              </div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function EvaluationView({ e }: { e: Evaluation }) {
  const armLabel = (a: string, f: number) => (a === "cleaned" ? "Cleaned" : `SMOTENC · ${f.toFixed(3)}`);
  const phrase = (f: Evaluation["findings"][number]) =>
    f.verdict === "mixed" ? `mixed across fill levels (${f.text})`
      : f.n > 1 ? `${f.verdict} at all ${f.n} fill levels (${f.text})` : `${f.verdict} (${f.text})`;
  return (
    <>
      <motion.div variants={rowV} className="evWrap">
        <table className="ev">
          <thead>
            <tr><th>Test set lift vs Original</th><th>Classifier</th>{e.metrics.map((m) => <th key={m}>{METRIC(m)}</th>)}</tr>
          </thead>
          <tbody>
            <tr className="ref">
              <td>Original (reference)</td>
              <td className="model">{e.reference_model}</td>
              {e.reference.map((v, i) => <td key={i}><span className="mean">{v.toFixed(3)}</span></td>)}
            </tr>
            {e.rows.map((r) => (
              <tr key={r.arm + r.fill}>
                <td>{armLabel(r.arm, r.fill)}</td>
                <td className="model">{r.model}</td>
                {r.cells.map((c, i) => (
                  <td key={i} className={tone(c.verdict)} title={c.verdict}>
                    {sign(c.lift)}<span className="mean">{c.mean.toFixed(3)}</span>
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        <p className="legend"><i className="gain" /> reliable gain <i className="loss" /> reliable loss <i className="flat" /> inconclusive · Classifier: picked per seed on validation, ×n = seeds that picked it</p>
      </motion.div>
      {(["cleaned", "smotenc"] as const).map((arm) => (
        <motion.div variants={rowV} className="finds" key={arm}>
          <h4>{arm === "cleaned" ? "Cleaned vs Original" : "SMOTENC vs Original"}</h4>
          {e.findings.filter((f) => f.arm === arm).map((f) => (
            <div className="find" key={f.metric}>
              <span>{METRIC(f.metric)}</span><b className={tone(f.verdict)}>{phrase(f)}</b>
            </div>
          ))}
        </motion.div>
      ))}
    </>
  );
}

function LogDropdown({ rows }: { rows: Decision[] }) {
  const [open, setOpen] = useState(false);
  if (!rows.length) return null;
  return (
    <div>
      <motion.button className="btn" style={{ marginTop: 10 }} whileTap={{ scale: 0.97 }} onClick={() => setOpen(!open)}>
        Decision log ({rows.length}) {open ? "−" : "+"}
      </motion.button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            style={{ overflow: "hidden" }}
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1, transition: { duration: 0.25, ease: EASE_OUT } }}
            exit={{ height: 0, opacity: 0, transition: { duration: 0.2 } }}
          >
            <motion.div
              initial="hidden"
              animate="show"
              variants={{ show: { transition: { staggerChildren: 0.03 } } }}
              style={{ maxHeight: "70vh", overflowY: "auto" }}
            >
              {rows.map((d, i) => (
                <motion.span className="row log" key={i} variants={rowV}>
                  <b>{label(d.rule)}</b> {short(d.outcome)}
                  <small>{d.rationale}</small>
                </motion.span>
              ))}
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

type DatasetInfo = { name: string; target: string; ready: boolean };

function StartScreen(props: {
  datasets: DatasetInfo[]; busy: boolean; starting: string | null; error: string | null;
  runs: string[]; onStart: (name: string) => void; onOpen: (run: string) => void;
}) {
  const { datasets, busy, starting, error, runs } = props;
  return (
    <main>
      <div className="glow" />
      <header>
        <div className="brand">Agentic GAN</div>
        {runs.length > 0 && <RunPicker runs={runs} value="Previous runs" onChange={props.onOpen} />}
      </header>
      <section className="hero">
        <div className="eyebrow">
          <span className={`live${starting ? " on" : ""}`} />
          {starting ? "Starting" : "New run"}
        </div>
        <div className="heroTitle">
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.h2
              key={starting ?? "choose"}
              initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.5, ease: EASE_OUT } }}
              exit={{ opacity: 0, y: -10, filter: "blur(8px)", transition: { duration: 0.4 } }}
            >
              {starting ? label(starting) : "Choose a dataset"}
            </motion.h2>
          </AnimatePresence>
        </div>
        <p className="sub">
          {error ?? (starting ? "The agents are getting started…" : busy ? "A run is already in progress." : "Pick one and the agents begin straight away.")}
        </p>
      </section>
      <div className="dsList" style={{ opacity: busy ? 0.4 : 1, transition: "opacity 0.25s" }}>
        {datasets.map((d, i) => (
          <motion.button
            key={d.name}
            className="ds"
            disabled={busy || !d.ready}
            whileTap={{ scale: 0.97 }}
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: d.ready ? 1 : 0.4, y: 0 }}
            transition={{ duration: IN, ease: EASE_OUT, delay: i * 0.03 }}
            onClick={() => props.onStart(d.name)}
          >
            <span className="num">{String(i + 1).padStart(2, "0")}</span>
            <span className="dsName">{label(d.name)}</span>
            <span className="dsMeta">{d.ready ? `Target: ${d.target}` : "Data file missing"}</span>
            <span className="dsGo">→</span>
          </motion.button>
        ))}
      </div>
    </main>
  );
}

function Icon({ status }: { status: string }) {
  return (
    <span className="icon">
      <AnimatePresence mode="popLayout" initial={false}>
        <motion.span
          key={status}
          style={{ position: "absolute", inset: 0, color: status === "SUCCESS" || status === "RUNNING" ? "var(--accent)" : "var(--muted)" }}
          initial={{ opacity: 0, filter: "blur(4px)" }}
          animate={{ opacity: 1, filter: "blur(0px)", transition: { duration: IN, ease: EASE_OUT } }}
          exit={{ opacity: 0, filter: "blur(4px)", transition: { duration: OUT } }}
        >
          {GLYPH[status]}
        </motion.span>
      </AnimatePresence>
    </span>
  );
}

export default function App() {
  const runs = useJson<string[]>("/api/runs") ?? [];
  const catalog = useJson<{ datasets: DatasetInfo[]; running: boolean }>("/api/datasets", 3000);
  const [picked, setPicked] = useState<string | null>(null);
  // The app opens on the dataset chooser; a run view follows once one is started or opened.
  const [mode, setMode] = useState<"start" | "run">("start");
  const [starting, setStarting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const newestBefore = useRef<string | null>(null);
  const runId = mode === "run" ? (picked ?? runs[0] ?? null) : null;
  const run = useJson<RunView>(runId && `/api/runs/${runId}`);
  const [selected, setSelected] = useState<string | null>(null);

  const nodeRefs = useRef<(HTMLButtonElement | null)[]>([]);
  const flowRef = useRef<HTMLDivElement>(null);
  const [centers, setCenters] = useState<number[]>([]);
  const [width, setWidth] = useState(0);
  const stageCount = run?.stages.length ?? 0;
  useLayoutEffect(() => {
    const measure = () => {
      setWidth(flowRef.current?.offsetWidth ?? 0);
      setCenters(nodeRefs.current.slice(0, stageCount).map((n) => (n ? n.offsetLeft + n.offsetWidth / 2 : 0)));
    };
    measure();
    window.addEventListener("resize", measure);
    return () => window.removeEventListener("resize", measure);
  }, [stageCount]);

  // A started run shows up as a new newest folder; open it as soon as it appears.
  useEffect(() => {
    if (starting && runs[0] && runs[0] !== newestBefore.current) {
      setPicked(runs[0]);
      setMode("run");
      setStarting(null);
    }
  }, [starting, runs]);

  const start = (name: string) => {
    setError(null);
    setStarting(name);
    newestBefore.current = runs[0] ?? null;
    fetch("/api/start", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset: name }) })
      .then((r) => { if (!r.ok) throw new Error(r.status === 409 ? "A run is already in progress." : "Could not start the run."); })
      .catch((e: Error) => { setStarting(null); setError(e.message); });
  };

  // Knob follows the pointer 1:1 while held, then snaps with ease-in-out on release.
  const knobX = useMotionValue(0);
  const knobScale = useMotionValue(1);
  const held = useRef(false);

  // Park the knob under the chosen (or current) stage whenever layout or progress changes.
  const knobIdx = selected && run ? run.stages.indexOf(selected) : run ? currentIndex(run) : 0;
  useEffect(() => {
    if (!held.current && centers[knobIdx] !== undefined) knobX.set(centers[knobIdx]);
  }, [centers, knobIdx, knobX]);

  if (mode === "start") {
    return (
      <StartScreen
        datasets={catalog?.datasets ?? []}
        busy={!!starting || !!catalog?.running}
        starting={starting}
        error={error}
        runs={runs}
        onStart={start}
        onOpen={(r) => { setPicked(r); setSelected(null); setMode("run"); }}
      />
    );
  }
  if (!runId || !run) {
    return (
      <main>
        <div className="brand">Agentic GAN</div>
        <p className="hint">Loading…</p>
      </main>
    );
  }

  const stages = run.stages;
  const status = (s: string) => stageStatus(run.state.stages[s]);
  const current = currentIndex(run);
  const selIdx = selected ? stages.indexOf(selected) : -1;
  const agentOf = (s: string) => run.agent_of[s];
  const retrying = run.decisions.filter((d) => d.agent === "model_selection" && d.rule === "type_routing").length > 1;
  const ready = centers.length === stages.length;

  const snap = (i: number) => {
    animate(knobX, centers[i] ?? 0, { duration: 0.35, ease: "easeInOut" });
    animate(knobScale, [1.12, 1], { duration: 0.35, ease: "easeInOut" });
  };

  const pick = (s: string) => {
    const next = selected === s ? null : s;
    setSelected(next);
    if (next) snap(stages.indexOf(next));
  };

  const detail = selected ? run.state.stages[selected] : undefined;
  const rows = selected ? run.decisions.filter((d) => d.agent === selected) : [];

  const allDone = stages.every((s) => ["SUCCESS", "SKIPPED"].includes(status(s)));
  const failed = stages.some((s) => status(s) === "FAILED");
  const heroStatus = failed ? "Failed" : allDone ? "Complete" : status(stages[current]) === "RUNNING" ? "Running" : "Waiting";
  const heroModel = run.models[stages[current]]?.split("/").pop();

  return (
    <main>
      <div className="glow" />
      <header>
        <div className="brand">Agentic GAN</div>
        <div className="headerRight">
          <motion.button className="btn" whileTap={{ scale: 0.97 }} onClick={() => setMode("start")}>New run</motion.button>
          <RunPicker runs={runs} value={runId} onChange={(r) => { setPicked(r); setSelected(null); }} />
        </div>
      </header>

      <section className="hero">
        <div className="eyebrow">
          <span className={`live${heroStatus === "Running" ? " on" : ""}`} />
          {heroStatus}
          {run.state.stages[stages[current]]?.attempts > 1 && ` · attempt ${run.state.stages[stages[current]].attempts}`}
        </div>
        <div className="heroTitle">
          <AnimatePresence mode="popLayout" initial={false}>
            <motion.h2
              key={current}
              initial={{ opacity: 0, y: 14, filter: "blur(8px)" }}
              animate={{ opacity: 1, y: 0, filter: "blur(0px)", transition: { duration: 0.5, ease: EASE_OUT } }}
              exit={{ opacity: 0, y: -10, filter: "blur(8px)", transition: { duration: 0.4 } }}
            >
              {label(stages[current])}
            </motion.h2>
          </AnimatePresence>
        </div>
        <p className="sub">
          {heroModel ? <>Powered by <b>{heroModel}</b></> : agentOf(stages[current]) ? "Rule-based step" : "Deterministic step"}
        </p>
      </section>

      <div className="flow" ref={flowRef}>
        <svg className="arcs">
          {ready && stages.slice(0, -1).map((s, i) => (
            <motion.path
              key={s}
              d={`M${centers[i]} 40 L${centers[i + 1]} 40`}
              stroke="#0297de" strokeWidth={1} fill="none"
              initial={false}
              animate={{ pathLength: i < current ? 1 : 0.001, opacity: i < current ? 1 : 0.2 }}
              transition={{ duration: 0.5, ease: "easeInOut" }}
            />
          ))}
          {ready && (
            <motion.path
              d={`M${centers[3]} 40 C${centers[3]} -4 ${centers[2]} -4 ${centers[2]} 40`}
              fill="none" strokeWidth={1} strokeDasharray="3 5"
              animate={{ stroke: retrying ? "#fc734d" : "rgba(255,255,255,0.14)" }}
              transition={{ duration: 0.4 }}
            />
          )}
        </svg>

        {ready && (
          <motion.div
            className="marker"
            style={{ position: "absolute", top: 34, left: 0, width: 12, height: 12, marginLeft: -6, zIndex: 2 }}
            initial={false}
            // x eases in-out while y rises and falls: the marker travels on an arc
            animate={{ x: centers[current], y: [0, -22, 0] }}
            transition={{ duration: 0.6, ease: "easeInOut" }}
          >
            {/* stretch 12% along the direction of travel */}
            <motion.div
              key={current}
              style={{ width: 12, height: 12, borderRadius: "50%", background: "var(--accent)", boxShadow: "0 0 18px 2px rgba(252,115,77,.55)" }}
              animate={{ scaleX: [1, 1.12, 1] }}
              transition={{ duration: 0.6, ease: "easeInOut" }}
            />
          </motion.div>
        )}

        <div className="nodes">
          {stages.map((s, i) => {
            const agent = agentOf(s);
            const model = run.models[s];
            return (
              <motion.button
                key={s}
                ref={(el) => { nodeRefs.current[i] = el; }}
                className={`node${i === current ? " current" : ""}`}
                whileTap={{ scale: 0.97 }}
                initial={{ opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: IN, ease: EASE_OUT, delay: i * 0.03 }}
                onClick={() => pick(s)}
              >
                {/* mask: a soft accent wash sweeps in when this stage becomes current */}
                <motion.span
                  className="fill"
                  initial={false}
                  animate={{ clipPath: i === current ? "inset(0 0% 0 0)" : "inset(0 100% 0 0)" }}
                  transition={{ duration: 0.5, ease: "easeInOut" }}
                />
                {status(s) === "SUCCESS" && (
                  <motion.span
                    className="pulse"
                    initial={{ opacity: 0.6, scale: 1 }}
                    animate={{ opacity: 0, scale: 1.06 }}
                    transition={{ duration: 0.6, ease: "easeOut" }}
                  />
                )}
                <div className="num">{String(i + 1).padStart(2, "0")}</div>
                <div className="name">{label(s)}</div>
                <div className="agent">{agent ? `${label(agent)} Agent` : "Deterministic"}</div>
                <div className={`chip${model ? "" : " none"}`}>{model?.split("/").pop() ?? (agent ? "Rules only" : "No model")}</div>
                <div className="state"><Icon status={status(s)} /><span>{status(s).toLowerCase()}</span></div>
                {selected === s && <motion.span layoutId="ring" className="ring" transition={{ duration: 0.3, ease: "easeInOut" }} />}
              </motion.button>
            );
          })}
        </div>
      </div>

      <div className="scrub">
        <div className="track" />
        <motion.div
          className="knob"
          style={{ x: knobX, scaleX: knobScale }}
          drag="x"
          dragConstraints={{ left: 0, right: width }}
          dragElastic={0.2}
          dragMomentum={false}
          onDragStart={() => { held.current = true; knobScale.set(1.12); }}
          onDragEnd={() => {
            held.current = false;
            const x = knobX.get();
            const i = centers.reduce((best, c, k) => (Math.abs(c - x) < Math.abs(centers[best] - x) ? k : best), 0);
            setSelected(stages[i]);
            snap(i);
          }}
        />
      </div>
      <p className="hint">Drag the knob or select a stage to see what happened</p>

      <div style={{ position: "relative" }}>
        <AnimatePresence mode="popLayout">
          {selected && detail && (
            <motion.div
              key={selected}
              className="popover"
              style={{ transformOrigin: `${centers[selIdx] ?? 0}px 0px` }}
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1, transition: { duration: IN, ease: EASE_OUT } }}
              exit={{ opacity: 0, scale: 0.95, transition: { duration: OUT } }}
            >
              <motion.div
                className="detail"
                initial="hidden"
                animate="show"
                variants={{ show: { transition: { delayChildren: 0.1, staggerChildren: 0.03 } } }}
              >
                <div className="detailHead">
                  <motion.div className="eyebrow" variants={rowV}>
                    {label(selected)} · {status(selected).toLowerCase()} · attempt {detail.attempts}
                  </motion.div>
                  {run.summaries[selected] && (
                    <motion.p className="headline" variants={rowV}>{run.summaries[selected].headline}</motion.p>
                  )}
                  {run.summaries[selected]?.explain && (
                    <motion.p className="explain" variants={rowV}>{run.summaries[selected].explain}</motion.p>
                  )}
                  {selected === "evaluation" && run.evaluation && (
                    <motion.div variants={rowV}><ReportDropdown sections={run.evaluation.sections} /></motion.div>
                  )}
                </div>
                <div className="detailBody">
                  {run.summaries[selected]?.gates?.map((g) => (
                    <motion.div className="gate" key={g.name} variants={rowV}>
                      <div className="gateHead">
                        <span>{label(g.name)}</span>
                        <b className={g.outcome === "PASS" ? "gain" : g.outcome === "FAIL" ? "loss" : "flat"}>{g.outcome.toLowerCase()}</b>
                      </div>
                      <p>{g.meaning}</p>
                      <small>{g.detail}</small>
                    </motion.div>
                  ))}
                  {run.summaries[selected]?.facts.map((f, i) => (
                    <motion.span className="row" key={i} variants={rowV}>{f}</motion.span>
                  ))}
                  {detail.reason && <motion.span className="row" variants={rowV}>{detail.reason}</motion.span>}
                  {selected === "evaluation" && run.evaluation && <EvaluationView e={run.evaluation} />}
                  <motion.div variants={rowV}><LogDropdown rows={rows} /></motion.div>
                </div>
              </motion.div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </main>
  );
}
