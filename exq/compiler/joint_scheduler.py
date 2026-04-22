"""Joint optimization for ExQ using CP-SAT.

This scheduler jointly assigns:
  - expert precision tiers
  - static prefetch decisions on high-probability edges

under memory and transfer-budget constraints.

It is intentionally small and pragmatic: the goal is to replace the fully greedy
view with one constrained optimization pass that can be compared against the
current heuristic pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

from ortools.sat.python import cp_model


PRECISIONS = ("BF16", "INT8", "INT4")
BIT_WIDTH = {"BF16": 16, "INT8": 8, "INT4": 4}
MEMORY_UNITS = {"BF16": 4, "INT8": 2, "INT4": 1}
ERROR_UNITS = {"BF16": 0, "INT8": 5, "INT4": 25}


@dataclass
class JointScheduleResult:
    quant_assignments: dict[tuple[int, int], str]
    prefetch_edges: list[tuple[int, int, int, int]]
    objective_value: float
    status: str


def solve_joint_schedule(
    graph,
    *,
    memory_budget_units: int | None = None,
    min_prefetch_prob: float = 0.35,
    max_prefetch_per_layer: int = 16,
    max_time_seconds: float = 30.0,
) -> JointScheduleResult:
    """Solve a joint quantization + prefetch schedule over the routing graph.

    `memory_budget_units` uses INT4 expert-size as the base unit. If omitted,
    it defaults to the size of an all-INT4 model plus a 10% slack budget.
    """
    hot_all = list(graph.hot_experts(0.0))
    if not hot_all:
        return JointScheduleResult({}, [], 0.0, "empty")

    nodes = [
        {"layer": layer, "expert": expert, "freq": float(freq)}
        for layer, expert, freq in hot_all
    ]
    node_index = {(n["layer"], n["expert"]): i for i, n in enumerate(nodes)}

    edges_raw = list(graph.high_prob_edges(min_prefetch_prob))
    edges = []
    per_layer_edge_count: dict[int, int] = {}
    for src_l, src_e, dst_l, dst_e, prob in edges_raw:
        layer_count = per_layer_edge_count.get(src_l, 0)
        if layer_count >= max_prefetch_per_layer:
            continue
        edges.append({
            "src_layer": src_l,
            "src_expert": src_e,
            "dst_layer": dst_l,
            "dst_expert": dst_e,
            "prob": float(prob),
        })
        per_layer_edge_count[src_l] = layer_count + 1

    baseline_units = len(nodes) * MEMORY_UNITS["INT4"]
    if memory_budget_units is None:
        memory_budget_units = int(baseline_units * 1.10)

    model = cp_model.CpModel()

    # Precision variable per node: 0=BF16, 1=INT8, 2=INT4
    pvars = [model.new_int_var(0, 2, f"p_{n['layer']}_{n['expert']}") for n in nodes]

    # Edge prefetch decisions
    evars = [
        model.new_bool_var(
            f"pf_{e['src_layer']}_{e['src_expert']}_{e['dst_layer']}_{e['dst_expert']}"
        )
        for e in edges
    ]

    # Memory budget
    mem_terms = []
    for pv in pvars:
        is_bf16 = model.new_bool_var(f"is_bf16_{pv.Name()}")
        is_int8 = model.new_bool_var(f"is_int8_{pv.Name()}")
        is_int4 = model.new_bool_var(f"is_int4_{pv.Name()}")
        model.add(pv == 0).only_enforce_if(is_bf16)
        model.add(pv != 0).only_enforce_if(is_bf16.Not())
        model.add(pv == 1).only_enforce_if(is_int8)
        model.add(pv != 1).only_enforce_if(is_int8.Not())
        model.add(pv == 2).only_enforce_if(is_int4)
        model.add(pv != 2).only_enforce_if(is_int4.Not())
        mem_terms.append(is_bf16 * MEMORY_UNITS["BF16"] + is_int8 * MEMORY_UNITS["INT8"] + is_int4 * MEMORY_UNITS["INT4"])
    model.add(sum(mem_terms) <= memory_budget_units)

    # Objective: minimize weighted quantization error and reward high-probability prefetches.
    # Scale to integers for CP-SAT.
    error_terms = []
    for pv, n in zip(pvars, nodes):
        freq_scaled = max(1, int(n["freq"] * 10000))
        is_bf16 = model.new_bool_var(f"err_bf16_{pv.Name()}")
        is_int8 = model.new_bool_var(f"err_int8_{pv.Name()}")
        is_int4 = model.new_bool_var(f"err_int4_{pv.Name()}")
        model.add(pv == 0).only_enforce_if(is_bf16)
        model.add(pv != 0).only_enforce_if(is_bf16.Not())
        model.add(pv == 1).only_enforce_if(is_int8)
        model.add(pv != 1).only_enforce_if(is_int8.Not())
        model.add(pv == 2).only_enforce_if(is_int4)
        model.add(pv != 2).only_enforce_if(is_int4.Not())
        error_terms.append(freq_scaled * (is_int8 * ERROR_UNITS["INT8"] + is_int4 * ERROR_UNITS["INT4"]))

    prefetch_bonus_terms = []
    for ev, e in zip(evars, edges):
        bonus = max(1, int(e["prob"] * 1000))
        prefetch_bonus_terms.append(ev * bonus)

    model.minimize(sum(error_terms) - sum(prefetch_bonus_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = max_time_seconds
    solver.parameters.num_search_workers = 8
    status = solver.solve(model)
    status_name = solver.StatusName(status)

    quant_assignments: dict[tuple[int, int], str] = {}
    for i, n in enumerate(nodes):
        quant_assignments[(n["layer"], n["expert"])] = PRECISIONS[solver.value(pvars[i])]

    selected_edges: list[tuple[int, int, int, int]] = []
    for ev, e in zip(evars, edges):
        if solver.value(ev):
            selected_edges.append((e["src_layer"], e["src_expert"], e["dst_layer"], e["dst_expert"]))

    return JointScheduleResult(
        quant_assignments=quant_assignments,
        prefetch_edges=selected_edges,
        objective_value=solver.objective_value,
        status=status_name,
    )
