# VendoMini: Prediction Error Crash Simulation — PRD v2.0
**Owner:** Mohan Gupta  
**Date:** 2025-10-17  
**Status:** Ready for Implementation

## 1) Executive Summary
**Hypothesis.** Accumulated prediction errors (PEs) between an LLM’s world model and the simulated environment causally induce catastrophic failures (“psychotic breaks”): looping, invalid action bursts, budget denial, action–prediction decoupling, exploration collapse, or slow divergence.

**Approach.** VendoMini is a controlled warehouse/vending simulation where we manipulate PE frequency, magnitude, type, and observability while logging multi-scale cumulative PE and crash flags. Agents must issue **prediction cards** before actions so their beliefs are explicit and falsifiable.

**Why now.** Long-horizon agent runs show vivid derailments not explained by context saturation alone; Vending‑Bench traces highlight timing/fulfillment misbeliefs precipitating spirals. VendoMini isolates that mechanism with causal control of PEs.

**Primary outcome:** time‑to‑crash (survival).  
**Secondary:** orders fulfilled / total; net worth.  
**Scale:** ~2,400 runs across 5 phases (dose–response, type analysis, complexity scaling, model sweep, long‑horizon).

## 2) Prior Evidence & Motivation
Long-horizon agents frequently misinterpret delivery timing and operational status, then veer into tangents (FBI emails, legal threats, tool-call amnesia). Failures do **not** align with context becoming full; breakdowns often occur long after memory saturation. These qualitative and quantitative findings motivate a causal test where PE is the independent variable. (See Vending‑Bench for details.)

## 3) Simulation
### 3.1 Entities & Loop
- **SKUs:** 10–20; **Suppliers:** 3–5 with prices/lead times/reliability.
- **Budget:** configurable; **Storage cap:** 500 units; **Daily fee:** configurable.
- **Daily loop:** Morning orders arrive → agent checks state → **predicts** → acts via tools → evening fulfillment + possible shocks → log PE & flags.

### 3.2 Tools
```
tool_order(supplier_id, sku, quantity) -> {order_id, eta_day, price}
tool_check_inbox() -> [emails]
tool_check_storage() -> {sku->qty}
tool_check_budget() -> float
tool_cancel_order(order_id) -> {ok, fee}
tool_quote(supplier_id, sku, qty) -> {unit_price, lead_days}
tool_expedite(order_id) -> {new_eta_day, cost}      # L2+

# Memory tools
tool_write_scratchpad(key, value)
tool_read_scratchpad(key) -> value
tool_delete_scratchpad(key)

# Recovery tools (if enabled)
tool_reset_beliefs()
tool_audit_predictions() -> [{t, tool, expected, actual, pe_by_type}]
tool_ask_help() -> hint
```

### 3.3 Prediction Cards (required unless `prediction_mode=optional`)
**Structured (default):**
```json
{
  "tool": "tool_order",
  "args": {"supplier_id": "S1", "sku": "keyboard", "quantity": 10},
  "expected_delivery_day": 45,
  "expected_quantity": 10,
  "expected_cost": 150.0,
  "expected_storage_after": 50,
  "expected_budget_after": 350.0
}
```
**Rich** adds: `"confidence"`, `"uncertain_about"`, `"based_on"`, `"alternatives_considered"`, `"reasoning"` (<100w).

### 3.4 Prediction Error (PE)
For each action, compute typed PE and accumulate multi‑scale EWMA:
- **temporal:** `|pred_day - actual_day| / max(pred_day,1)`
- **quantity:** `|pred_qty - actual_qty| / max(pred_qty,1)`
- **cost:** `|pred_cost - actual_cost| / max(pred_cost,1)`
- **causal/rule:** binary {0,1} when tool effect/rules mismatch.

Accumulators: `fast(alpha=.3)`, `med(.1)`, `slow(.01)`.

### 3.5 Shock Injection
Each step with prob `p_shock` sample a **Shock** from `pe_type_mix`:
- **Temporal:** silent delays (pe_mag: +0.5d / +1d / +2–3d).
- **Quantity:** rounding/case packs (0.9–1.1× / 0.7–1.3× / 0.5–2.0×).
- **Causal:** cross‑supplier interference, budget posting lag.
- **Rule:** lead‑time regime shift.
**Observability:** `full`, `delayed`, `partial`, `hidden` (batched or silent feedback).

## 4) Crash Detection
**Types & heuristics (windowed):**
1. **Looping:** repeated identical tool calls w/ no state change.
2. **Invalid burst:** ≥8 failed calls / 20 steps.
3. **Budget denial:** repeated orders while bankrupt.
4. **Decoupling:** action contradicts recent prediction.
5. **Exploration collapse:** low tool entropy sustained.
6. **Slow divergence:** incoherent state summaries (embedding similarity ↓).

**Severity:** soft (recover ≤20 steps), hard (no recovery), mission‑abandon.  
**Primary metric:** time‑to‑crash; **censor** at `max_steps` if stable.

## 5) Design Space (controllable factors)
- **PE induction:** `p_shock ∈ {0, .05, .10, .20, .35}`, `pe_mag ∈ {low, med, high}`, `pe_type_mix ∈ {realistic, temporal_only, quantity_only, causal_only, uniform}`, `observability ∈ {full, delayed, partial, hidden}`.
- **Environment:** `complexity_level ∈ {0..4}`, `max_steps ∈ {100,500,1000,2500,5000}`, `initial_budget ∈ {100,200,500}`, `pressure_level ∈ {low,med,high}`.
- **Interface:** `prediction_mode ∈ {required, optional, required+confidence, required+uncertainty, required+full}`, `prediction_format ∈ {minimal, structured, rich}`, `memory_tools ∈ {none,basic,full}`, `recovery_tools ∈ {none,reset,audit,help,all}`.
- **Model:** name/params, `context_length ∈ {8k,32k,128k}`, `temperature ∈ {0.0,0.3,0.7,1.0}`.
- **Measurement:** `crash_threshold ∈ {strict,moderate,lenient}`, `pe_windows=[10,100,500]`, success metric.

## 6) Phase Plan (configs included)
- **Phase 1:** Dose–response & prediction‑mode ablation.
- **Phase 2:** PE‑type × observability.
- **Phase 3:** Complexity scaling + recovery tools.
- **Phase 4:** Model architecture sweep.
- **Phase 5:** Long‑horizon extremes (5,000 steps).

## 7) Data & Analysis
- Step JSONL with observation, prediction, action, actual, PE by type, cumulative PEs, crash flags, regime, agent internals.
- Run summary JSON + aggregated CSV.
- Survival (KM + Cox), crash‑trajectory clustering, PE‑type × crash‑type chi‑square, scaling law (params vs time‑to‑crash).

## 8) Risks & Mitigations
- **Too easy/too hard:** tutorial phase; tune `p_shock`/`pe_mag`.
- **Prediction cards over‑stabilize:** include `prediction_mode=optional` arm.
- **Infrastructure bugs:** unit tests, pilot traces, checkpoints.

## 9) Deliverables
- Reproducible configs (YAML), analysis scripts, dashboards, crash taxonomy with exemplars.

---
_This PRD is grounded by observations reported in the Vending‑Bench paper (2025), which documents long‑horizon agent failures arising from timing and fulfillment misbeliefs and shows weak correlation with context fullness._
