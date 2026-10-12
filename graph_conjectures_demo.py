#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["networkx>=3.3", "numpy>=1.26", "pandas>=2.0"]
# ///
"""A conjecture-discovery loop on small graphs, in about 200 lines.

    data -> invariants -> conjectures -> triage -> new experiments -> re-check

Round 1  a deliberately biased "lab" (dense random graphs only); a tiny
         Graffiti-style engine proposes bounds on max-cut.
Round 2  the experiments a reviewer would ask for (sparse graphs, trees,
         cycles, cliques); dead conjectures get repaired.
Round 3  one adversarial experiment aimed at the survivors.

Run:  python graph_conjectures_demo.py [--seed 7]   (about 10 s, writes triage_prompt.md)
"""
import argparse
import itertools
import math
import random
from fractions import Fraction as F

import networkx as nx
import numpy as np
import pandas as pd

TARGET = "maxcut"
INVARIANTS = ["n", "m", "min_deg", "max_deg", "alpha", "matching", "triangles"]
HYPOTHESES = ["all", "bipartite", "non_bipartite"]
COEFFS = [F(-1), F(-1, 2), F(-1, 4), F(1, 4), F(1, 2), F(3, 4), F(1), F(3, 2), F(2)]


# ---------------------------------------------------------------- 1. data
def max_cut(G):
    """Exact max-cut by brute force over all 2^(n-1) bipartitions (n <= 14)."""
    idx = {v: i for i, v in enumerate(G)}
    E = np.array([(idx[u], idx[v]) for u, v in G.edges()])
    n = G.number_of_nodes()
    sides = (np.arange(2 ** (n - 1))[:, None] >> np.arange(n - 1)) & 1
    sides = np.hstack([np.zeros((len(sides), 1), dtype=int), sides])
    return int((sides[:, E[:, 0]] != sides[:, E[:, 1]]).sum(axis=1).max())


def invariants(G, family):
    return dict(
        family=family,
        n=G.number_of_nodes(),
        m=G.number_of_edges(),
        min_deg=min(d for _, d in G.degree()),
        max_deg=max(d for _, d in G.degree()),
        alpha=max(len(c) for c in nx.find_cliques(nx.complement(G))),  # independence number
        matching=len(nx.max_weight_matching(G, maxcardinality=True)),
        triangles=sum(nx.triangles(G).values()) // 3,
        maxcut=max_cut(G),
        all=True,
        bipartite=nx.is_bipartite(G),
    )


def table(rows):
    df = pd.DataFrame(rows)
    df["non_bipartite"] = ~df["bipartite"]
    return df


def connected_gnp(n, p, rng):
    while True:
        G = nx.gnp_random_graph(n, p, seed=rng.randrange(10**9))
        if nx.is_connected(G):
            return G


def round1_lab(rng):
    """The biased lab: only dense random graphs."""
    return table(invariants(connected_gnp(rng.randint(6, 10), rng.uniform(0.5, 0.9), rng), "dense random")
                 for _ in range(150))


def round2_reviewer(rng):
    """What a reviewer would ask for: sparse graphs, trees, cycles, cliques."""
    rows = []
    for _ in range(60):
        n = rng.randint(6, 12)
        rows.append(invariants(connected_gnp(n, rng.uniform(0.15, 0.35), rng), "sparse random"))
        rows.append(invariants(nx.random_labeled_tree(n, seed=rng.randrange(10**9)), "tree"))
    for n in range(4, 13):
        rows.append(invariants(nx.cycle_graph(n), f"cycle C{n}"))
        rows.append(invariants(nx.complete_graph(n), f"clique K{n}"))
        rows.append(invariants(nx.complete_bipartite_graph(n // 2, n - n // 2), f"K{n // 2},{n - n // 2}"))
    return table(rows)


def round3_adversarial():
    """Aimed at the survivors: dense graphs that are ONE edge away from bipartite."""
    rows = []
    for a, b in [(3, 3), (3, 4), (4, 4), (4, 5), (5, 5)]:
        G = nx.complete_bipartite_graph(a, b)
        G.add_edge(0, 1)  # one edge inside a part creates odd cycles
        rows.append(invariants(G, f"K{a},{b} + 1 edge"))
    return table(rows)


# ---------------------------------------------------- 2. conjecture engine
def evaluate(df, c):
    """Right-hand side of conjecture c on every graph in its hypothesis class."""
    sub = df[df[c["hyp"]]]
    rhs = sub[list(c["feats"])].to_numpy(dtype=float) @ np.array([float(a) for a in c["coeffs"]])
    return sub, rhs + float(c["c"])


def fit(df, hyp, feats, coeffs, sense):
    """Tightest constant c so that `target sense rhs + c` holds on every graph in the class."""
    sub = df[df[hyp]]
    t = sub[TARGET].to_numpy(dtype=float)
    rhs = sub[list(feats)].to_numpy(dtype=float) @ np.array([float(a) for a in coeffs])
    c = F(math.floor(4 * (t - rhs).min()), 4) if sense == ">=" else F(math.ceil(4 * (t - rhs).max()), 4)
    bound = rhs + float(c)
    return dict(hyp=hyp, feats=feats, coeffs=coeffs, sense=sense, c=c, bound=bound,
                touch=int(np.isclose(t, bound).sum()), size=len(sub))


def generate_conjectures(df, top_k, min_class=10, min_touch=3):
    """Graffiti-style: enumerate small linear bounds, keep sharp, non-redundant, simple ones."""
    kept, n_candidates, seen_classes = [], 0, set()
    for hyp in HYPOTHESES:
        members = tuple(df.index[df[hyp]])
        if len(members) < min_class:
            print(f"  (no conjectures about '{hyp}' graphs: the data has {len(members)})")
            continue
        if members in seen_classes:  # same graphs as an earlier class: nothing new to say
            continue
        seen_classes.add(members)
        for sense in (">=", "<="):
            cands = [fit(df, hyp, feats, coeffs, sense)
                     for k in (1, 2)
                     for feats in itertools.combinations(INVARIANTS, k)
                     for coeffs in itertools.product(COEFFS, repeat=k)]
            n_candidates += len(cands)
            cands = [c for c in cands if c["touch"] >= min_touch]
            cands.sort(key=lambda c: (-c["touch"], len(c["feats"]), abs(c["c"])))
            # Dalmatian heuristic: keep a bound only if it is the best bound on some graph.
            best, accepted = None, []
            for cand in cands:
                b = cand["bound"]
                if best is None or (np.any(b > best + 1e-9) if sense == ">=" else np.any(b < best - 1e-9)):
                    accepted.append(cand)
                    best = b if best is None else (np.maximum(best, b) if sense == ">=" else np.minimum(best, b))
            kept += accepted[:top_k]
    return kept, n_candidates


def show(c):
    terms = [f"{'-' if a < 0 else '+'} {'' if abs(a) == 1 else str(abs(a)) + '*'}{x}"
             for a, x in zip(c["coeffs"], c["feats"])]
    if c["c"] != 0:
        terms.append(f"{'-' if c['c'] < 0 else '+'} {abs(c['c'])}")
    return f"[{c['hyp']}] {TARGET} {c['sense']} {' '.join(terms).lstrip('+ ')}"


# ------------------------------------------------- 3. re-check and repair
def counterexample(c, df):
    """Worst violation of conjecture c on df, or None if it holds everywhere."""
    sub, bound = evaluate(df, c)
    if sub.empty:
        return None
    gap = (bound - sub[TARGET]) if c["sense"] == ">=" else (sub[TARGET] - bound)
    i = int(np.argmax(gap.to_numpy()))
    if gap.iloc[i] <= 1e-9:
        return None
    row = sub.iloc[i]
    return f"{row.family}: maxcut = {row.maxcut}, bound says {bound[i]:g}"


def repair(c, df):
    """Mutation operator. First try narrowing the hypothesis, then weakening the constant."""
    for hyp in HYPOTHESES:
        if hyp != c["hyp"] and df[hyp].any() and set(df.index[df[hyp]]) < set(df.index[df[c["hyp"]]]):
            narrowed = dict(c, hyp=hyp)
            if counterexample(narrowed, df) is None:
                return "add hypothesis", narrowed
    return "weaken constant", fit(df, c["hyp"], c["feats"], c["coeffs"], c["sense"])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--seed", type=int, default=7, help="random seed for the graph samples (default 7)")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    pd.set_option("display.width", 120)
    bar = "=" * 78

    print(f"{bar}\nROUND 1  biased lab: 150 dense random graphs, n = 6..10\n{bar}")
    data = round1_lab(rng)
    print(data[["family", *INVARIANTS, TARGET]].head(4).to_string(index=False))
    print(f"  bipartite graphs in the data: {int(data.bipartite.sum())}")
    conjs, n_cand = generate_conjectures(data, top_k=4)
    print(f"\n  engine scored {n_cand:,} candidate bounds; showing the sharpest {len(conjs)}:")
    for i, c in enumerate(conjs, 1):
        print(f"  C{i}  {show(c):48s} tight on {c['touch']}/{c['size']}")

    with open("triage_prompt.md", "w", encoding="utf-8") as f:
        f.write("You are reviewing machine-generated conjectures about the max-cut of a graph.\n"
                "The data was ONLY dense connected random graphs with n = 6..10.\n\n")
        f.writelines(f"C{i}: {show(c)}   (tight on {c['touch']}/{c['size']} graphs)\n"
                     for i, c in enumerate(conjs, 1))
        f.write("\nFor each: label TRIVIAL / KNOWN (name the theorem) / SUSPICIOUS (likely a sampling "
                "artifact) / INTERESTING, in one line. Then name the graph families to sample next "
                "that would most likely falsify the suspicious ones.\n")
    print("  -> wrote triage_prompt.md  (the LLM critic + experiment-planner step)")

    print(f"\n{bar}\nROUND 2  reviewer's experiments: sparse graphs, trees, cycles, cliques, K_a,b\n{bar}")
    new = round2_reviewer(rng)
    data = pd.concat([data, new], ignore_index=True)
    survivors, killed, repaired = [], 0, 0
    for i, c in enumerate(conjs, 1):
        cx = counterexample(c, new)
        if cx is None:
            print(f"  C{i}  survives     {show(c)}")
            survivors.append(c)
            continue
        killed += 1
        how, fixed = repair(c, data)
        print(f"  C{i}  KILLED by {cx}")
        if how == "add hypothesis":
            print(f"      repair ({how}) -> {show(fixed)}")
            survivors.append(fixed)
            repaired += 1
    print(f"\n  {len(conjs)} tested: {killed} killed as stated, {repaired} rescued by a narrower hypothesis")

    print(f"\n{bar}\nROUND 3  adversarial experiment: dense graphs one edge away from bipartite\n{bar}")
    adv = round3_adversarial()
    data = pd.concat([data, adv], ignore_index=True)
    for c in survivors:
        if not adv[c["hyp"]].any():
            print(f"  untested    {show(c):42s} (no {c['hyp']} graphs in this experiment)")
            continue
        cx = counterexample(c, adv)
        print(f"  {'survives   ' if cx is None else 'KILLED     '} {show(c):42s} {'' if cx is None else 'by ' + cx}")

    print(f"\n{bar}\nWHAT THE LOOP NOW BELIEVES (re-mined on all {len(data)} graphs)\n{bar}")
    final, _ = generate_conjectures(data, top_k=2)
    for c in final:
        print(f"  {show(c):48s} tight on {c['touch']}/{c['size']}")


if __name__ == "__main__":
    main()
