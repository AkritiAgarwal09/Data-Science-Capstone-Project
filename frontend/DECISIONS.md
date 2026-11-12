# Frontend decisions

Status: built. Vite + React + TypeScript + Motion (`motion/react`), chosen from the options below.

## Decisions since

- One application: `src/agentic_gan/web.py` (stdlib `http.server`, no new Python dependency) serves `frontend/dist` and a read-only API (`/api/runs`, `/api/runs/<id>`). Run it with `npm --prefix frontend run build` then `uv run python -m agentic_gan.web [port]`. For frontend work, `npm --prefix frontend run dev` proxies `/api` to port 8000.
- Live updates by polling every 2 s.
- Model per agent: `run_llm_step` now records the model in the `llm_step` decision's inputs; the API reports the latest one per agent. Agents with no LLM decision show "rules only"; split shows "no model".
- Font: Inter everywhere, self-hosted via `@fontsource-variable/inter` (no CDN).
- Each stage's popover shows a headline and bullet facts (split ratios and row counts, preprocessing treatments, chosen generator and gates, quality verdict and diagnosis, and the training and evaluation summaries), with the raw decision log in a dropdown below. `web.py` builds these from the run's own files, with no extra LLM calls.
- The app opens on a dataset chooser. Picking one POSTs `/api/start`; `web.py` launches `python -m agentic_gan.orchestrator --dataset <name>` as a subprocess (loading `.env`, one run at a time, JSON content type required) and the page opens the new run as soon as its folder appears. Progress prints in the server terminal. Datasets whose data file is missing are shown disabled.
- Model selection and quality check explanations come from the agents' own instructions (`EXPLAIN_PROMPT`, `DIAGNOSIS_PROMPT`), logged in `decisions.jsonl` and shown under each stage's headline. Quality check only explains when the gate fails, because it only calls the LLM then.
- Flow diagram is hand-drawn SVG, as recommended. No React Flow.

## Goal

Show the agent pipeline as a flow, mark where the current run is, and show which model each agent uses.

Stages come from `Stage` in `src/agentic_gan/schemas.py`, in run order:
`split → preprocessing → model_selection → quality_check → training → evaluation`.
Quality check can send the run back to model selection (retry loop) or skip training (`gan_available = false`).
Stage state (`status`, `attempts`, `running`, `reason`) lives in `runs/<run_id>/state.json`.
Agent decisions are logged in `runs/<run_id>/decisions.jsonl`.

## Data

- Read `state.json` and `decisions.jsonl` for a run. No backend logic is added to the pipeline.
- Open question: the LLM model is one global `VERTEX_MODEL` env var (`llm/vertex.py`), so "model per agent" has no per-agent source yet. Options: (a) show the global model for every LLM agent and "rules only" for non-LLM stages, (b) have `run_llm_step` record the model in each `decisions.jsonl` row. Recommend (b), a one-line change.
- Transport: a tiny read-only server (FastAPI or static file serve) with polling, upgrade to SSE if polling feels laggy.

## Palette

| Role | Hex |
|------|-----|
| Primary (midnight black) | `#02070A` |
| Secondary (electric blue) | `#0297DE` |
| Accent (cantaloupe orange) | `#fc734d` |

Primary is the page and surface colour. Secondary is for flow lines, nodes and links.
Accent is for the current stage, retries and success pulses.

## Motion rules (as given)

Entrances and exits
- Ease-out for entrances.
- Exits 20% shorter than the matching entrance.
- Overlap old and new states (no gap between exit and enter).
- Dropdowns 150–250 ms.
- Popovers start at scale 0.95, enter from the trigger, exit to the trigger.
- No bounce on everyday UI.

Lists and details
- Stagger list rows by 30 ms.
- Stagger details as they settle.

Movement
- Ease-in-out for moves.
- Arcs for object movement (e.g. the "current" marker travelling between stages follows a curve, not a straight line).
- Stretch moving knobs by 12% along the direction of travel.
- Rubber band when dragged past bounds.
- Do not tween while held (follow the pointer 1:1 while dragging; animate only on release).

Feedback
- Scale to 0.97 on press.
- Pulse to confirm success.
- Crossfade icons with blur.
- Masks for reveals and colour changes.
- Reuse the same `layoutId` for an element that moves between places, so it animates instead of remounting.

## Where each rule applies

| Rule | Place in this UI |
|------|------------------|
| Arc + ease-in-out + stretch 12% | "You are here" marker moving between stage nodes |
| Reuse `layoutId` | The marker, and a model chip that moves with its agent when selected |
| Stagger 30 ms | Decision log rows, stage list |
| Stagger as details settle | Agent detail panel fields after it opens |
| Popover 0.95, from/to trigger | Agent detail popover opened from a node |
| Dropdown 150–250 ms | Run picker |
| Press 0.97 | All buttons and nodes |
| Pulse | Stage turning SUCCESS |
| Crossfade + blur | Status icon changing (running → success / failed / retry) |
| Masks | Node fill sweeping to accent when it becomes current |
| Rubber band + no tween while held | Draggable run-history scrubber (replays a past run) |
| Overlap old/new | Switching runs, swapping detail panel content |

## Framework options

Needs: `layoutId`-style shared layout animation, exit animations, drag with elastic bounds, path/arc motion, small data-driven UI.

| Option | Fit | Notes |
|--------|-----|-------|
| **Vite + React + TypeScript + Motion (`motion/react`)** — recommended | Best | `layoutId`, `AnimatePresence`, `dragElastic`, keyframe arcs and `stagger` are built in. Maps one-to-one to the list above. |
| Vite + Vue + `motion-v` | Good | Port of the same API, smaller community. |
| SvelteKit + `svelte/motion` | Partial | Nice springs and transitions; no `layoutId` equivalent (FLIP by hand). |
| Next.js + Motion | Works, overkill | SSR and routing are not needed for a local dashboard. |

Supporting picks (recommended stack):
- Flow diagram: hand-drawn SVG. Six nodes and one loop do not justify React Flow, and arcs/masks are easier on plain SVG.
- Styling: Tailwind v4 with the palette as CSS variables.
- Tests: Vitest for the state-to-view mapping.

## Notes

- No "motion skill" is installed in this environment. The motion rules above are taken from your list directly.
- Respect `prefers-reduced-motion` (fade only), unless told otherwise.
