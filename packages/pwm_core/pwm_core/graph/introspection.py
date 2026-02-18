"""pwm_core.graph.introspection
================================

Deterministic text explanation of graph structure, nodes, edges,
parameter counts -- without LLM.

Functions
---------
explain_graph   Produce a multi-line text summary of a compiled GraphOperator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pwm_core.graph.graph_operator import GraphOperator


def explain_graph(graph_op: "GraphOperator") -> str:
    """Return a deterministic text explanation of the graph structure.

    Includes:
    * Graph ID and overall properties (linear, node/edge counts).
    * Per-node detail: primitive type, parameter count, linearity.
    * Edge list.
    * Learnable parameters summary.
    * Total parameter count (scalar + blob elements).

    Parameters
    ----------
    graph_op : GraphOperator
        A compiled graph operator.

    Returns
    -------
    str
        Multi-line human-readable explanation.
    """
    lines: list[str] = []
    lines.append(f"OperatorGraph: {graph_op.graph_id}")
    lines.append("=" * (len(lines[0])))
    lines.append("")

    # Summary
    lines.append("Summary")
    lines.append("-------")
    lines.append(f"  Nodes:       {len(graph_op.forward_plan)}")
    lines.append(f"  Edges:       {len(graph_op.edges)}")
    lines.append(f"  All linear:  {graph_op.all_linear}")
    lines.append(f"  x_shape:     {graph_op.x_shape}")
    lines.append(f"  y_shape:     {graph_op.y_shape}")
    lines.append("")

    # Forward plan
    lines.append("Forward execution plan (topological order)")
    lines.append("------------------------------------------")
    total_scalar_params = 0
    total_blob_elements = 0

    for i, (node_id, prim) in enumerate(graph_op.forward_plan):
        ser = prim.serialize()
        n_params = len(ser.get("params", {}))
        n_blobs = len(ser.get("blobs", []))
        blob_elements = sum(
            _product(b.get("shape", [])) for b in ser.get("blobs", [])
        )
        total_scalar_params += n_params
        total_blob_elements += blob_elements

        linear_tag = "linear" if prim.is_linear else "non-linear"
        lines.append(
            f"  {i + 1}. [{node_id}] primitive={prim.primitive_id} "
            f"({linear_tag}, {n_params} scalar params, {n_blobs} blobs)"
        )

    lines.append("")

    # Edges
    if graph_op.edges:
        lines.append("Edges")
        lines.append("-----")
        for src, tgt in graph_op.edges:
            lines.append(f"  {src} -> {tgt}")
        lines.append("")

    # Learnable params
    if graph_op.learnable_params:
        lines.append("Learnable parameters")
        lines.append("--------------------")
        for nid, param_names in graph_op.learnable_params.items():
            lines.append(f"  {nid}: {', '.join(param_names)}")
        lines.append("")

    # Totals
    lines.append("Parameter totals")
    lines.append("----------------")
    lines.append(f"  Scalar params:   {total_scalar_params}")
    lines.append(f"  Blob elements:   {total_blob_elements}")
    total_learnable = sum(
        len(v) for v in graph_op.learnable_params.values()
    )
    lines.append(f"  Learnable count: {total_learnable}")
    lines.append("")

    return "\n".join(lines)


def _product(shape: list) -> int:
    """Compute the product of a shape list."""
    result = 1
    for s in shape:
        result *= s
    return result
