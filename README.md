# Agentic Conjecture Loop Demo

A laptop-sized agentic discovery loop in one Python file (about 200 lines):
generate data, compute invariants, propose conjectures, triage them with an LLM,
run new experiments to try to refute them (find a counterexample), and repair the ones that die.

Companion code for the tutorial *Agents That Conjecture* (Agentic Development & Discovery, Fall 2026).
The full research version of this loop is **SCALAR**: [arXiv:2605.10327](https://arxiv.org/abs/2605.10327) · [code](https://github.com/sfeeney1897/SCALAR).

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sfeeney1897/agentic-conjecture-loop-demo/blob/main/demo.ipynb)

## Run it

Pick whichever option fits your machine. All of them work on Windows, macOS and Linux. No GPU needed; it runs in about 10 seconds.

**No git?** Use the green **Code → Download ZIP** button on GitHub, unzip, and open a terminal in that folder.

### Option A: uv (one command, installs everything for you)

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) once, then:

```
git clone https://github.com/sfeeney1897/agentic-conjecture-loop-demo.git
cd agentic-conjecture-loop-demo
uv run graph_conjectures_demo.py
```

### Option B: plain Python 3.10+ and pip

macOS / Linux:

```
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python graph_conjectures_demo.py
```

Windows (PowerShell or Command Prompt):

```
py -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python graph_conjectures_demo.py
```

### Option C: nothing installed

Click **Open in Colab** above and run the cells.

## What you will see

The target is the **max-cut** of a graph (computed exactly by brute force). The vocabulary is seven cheap invariants:
vertices `n`, edges `m`, `min_deg`, `max_deg`, independence number `alpha`, `matching` number, and `triangles`.

| Round | Experiment | What happens (default seed 7) |
|---|---|---|
| 1 | 150 dense random graphs | Zero of them are bipartite. The engine scores 3,528 candidate bounds and keeps 8, including `maxcut <= m - 1`. |
| 2 | Trees, cycles, cliques, complete bipartite graphs | All 8 die as stated. A tree kills `maxcut <= m - 1`; the repair narrows it to **non-bipartite graphs**, which is a true theorem (an odd cycle always leaves one edge uncut). |
| 3 | Dense graphs one edge away from bipartite | A repaired survivor, `non-bipartite: maxcut <= m - min_deg + 1`, dies on K5,5 plus one edge. |

The lesson: **true on your data is not true.** Surviving more data is not proof, which is why serious systems end with a proof checker such as Lean.

## The LLM step

Round 1 writes `triage_prompt.md`. Paste it into any LLM (Claude, ChatGPT, ...) and it will label each conjecture
as trivial, known, suspicious or interesting, and suggest which graphs to sample next. Compare its suggestions with rounds 2 and 3.

## How it maps to the course

| Submodule | In the code | Course idea |
|---|---|---|
| Experimenter | `round1_lab`, `round2_reviewer`, `round3_adversarial` | the data behind the harness (Ch 7) |
| Conjecturer | `generate_conjectures` | Generate (Ch 9) |
| Critic and planner | `triage_prompt.md` + an LLM | LLM as judge (Ch 7.5), context engineering (Ch 4) |
| Refuter | `counterexample` | Evaluate / fitness (Ch 10.4) |
| Mutator | `repair` | mutation operator (Ch 10.2) |
| Selector | the "best on some graph" filter in `generate_conjectures` | selection and diversity (Ch 10.3, 10.5) |
| Verifier | not included | the oracle end (Ch 7.5); see SCALAR + Lean |

## Try it yourself

1. Run with `--seed 11` (or any number). Which conjectures change, and which are stable?
2. Add an invariant (diameter, chromatic number, ...) to `INVARIANTS` and `invariants()`. Does the engine find new laws?
3. Change `TARGET` from `maxcut` to `alpha`.
4. Make the planner real: call an LLM API with `triage_prompt.md` and build round 2 from the graph families it suggests.
5. Make it evolutionary: keep running rounds until no conjecture dies, and log the population each generation.
6. Point it at your own research data: rows are objects, columns are invariants, pick a target.

## Background

- Graffiti (Fajtlowicz, 1980s) and TxGraffiti (Davila) pioneered data-driven conjecture generation in graph theory; the "best on some graph" filter is a simplified version of Graffiti's Dalmatian heuristic.
- SCALAR applies the same loop to quantum circuits (QAOA on MaxCut, simulated with CUDA-Q).
