"""pwm_core.graph.canonical
============================

Canonical chain validator for the universal forward model.

Enforces: Source -> Element(s) -> Sensor -> Noise topology.

The validator is called from ``GraphCompiler.compile()`` when a graph spec
has ``metadata.canonical_chain = True``.

Role detection priority:
1. ``node.role`` field (explicit)
2. ``node.tags.node_role`` (from NodeTags)
3. Inferred from ``primitive_id`` via ``_node_role`` class attribute on primitive
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Dict, List, Optional, Set, Tuple

from pwm_core.graph.compiler import GraphCompilationError
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.ir_types import NodeRole
from pwm_core.graph.primitives import PRIMITIVE_REGISTRY

logger = logging.getLogger(__name__)


def _get_node_role(node, spec: OperatorGraphSpec) -> Optional[NodeRole]:
    """Determine the role of a node using the 3-level priority chain."""
    # 1. Explicit role field on GraphNode
    if node.role is not None:
        return NodeRole(node.role)

    # 2. tags.node_role
    if node.tags is not None and node.tags.node_role is not None:
        return NodeRole(node.tags.node_role)

    # 3. Infer from primitive class attribute
    prim_cls = PRIMITIVE_REGISTRY.get(node.primitive_id)
    if prim_cls is not None:
        role_str = getattr(prim_cls, "_node_role", None)
        if role_str is not None:
            return NodeRole(role_str)

    return None


def validate_canonical_chain(spec: OperatorGraphSpec) -> None:
    """Validate that the graph follows Source -> Element(s) -> Sensor -> Noise.

    Raises
    ------
    GraphCompilationError
        If the canonical chain is violated.
    """
    # Build role map
    role_map: Dict[str, Optional[NodeRole]] = {}
    for node in spec.nodes:
        role_map[node.node_id] = _get_node_role(node, spec)

    # Classify nodes by role
    sources = [nid for nid, r in role_map.items() if r == NodeRole.source]
    sensors = [nid for nid, r in role_map.items() if r == NodeRole.sensor]
    noises = [nid for nid, r in role_map.items() if r == NodeRole.noise]
    corrections = [nid for nid, r in role_map.items() if r == NodeRole.correction]
    elements = [
        nid for nid, r in role_map.items()
        if r not in (NodeRole.source, NodeRole.sensor, NodeRole.noise, NodeRole.correction)
    ]

    # Check: exactly 1 source
    if len(sources) != 1:
        raise GraphCompilationError(
            f"Canonical chain requires exactly 1 source node, "
            f"found {len(sources)}: {sources}"
        )

    # Check: at most 1 correction node
    if len(corrections) > 1:
        raise GraphCompilationError(
            f"Canonical chain allows at most 1 correction node, "
            f"found {len(corrections)}: {corrections}"
        )

    # Check: at least 1 element (correction nodes do NOT count)
    if len(elements) < 1:
        raise GraphCompilationError(
            f"Canonical chain requires at least 1 element/transport node, "
            f"found {len(elements)}: {elements}"
        )

    # Check: exactly 1 sensor
    if len(sensors) != 1:
        raise GraphCompilationError(
            f"Canonical chain requires exactly 1 sensor node, "
            f"found {len(sensors)}: {sensors}"
        )

    # Check: exactly 1 noise
    if len(noises) != 1:
        raise GraphCompilationError(
            f"Canonical chain requires exactly 1 noise node, "
            f"found {len(noises)}: {noises}"
        )

    source_id = sources[0]
    sensor_id = sensors[0]
    noise_id = noises[0]

    # Build adjacency and reverse adjacency
    adj: Dict[str, List[str]] = defaultdict(list)
    in_adj: Dict[str, List[str]] = defaultdict(list)
    for edge in spec.edges:
        adj[edge.source].append(edge.target)
        in_adj[edge.target].append(edge.source)

    # Check: Noise is a sink (no outgoing edges)
    if adj.get(noise_id):
        raise GraphCompilationError(
            f"Noise node '{noise_id}' must be a sink (no outgoing edges), "
            f"but has edges to: {adj[noise_id]}"
        )

    # Check: Sensor -> Noise edge exists
    if noise_id not in adj.get(sensor_id, []):
        raise GraphCompilationError(
            f"Canonical chain requires edge from sensor '{sensor_id}' "
            f"to noise '{noise_id}'"
        )

    # Check: directed path Source -> ... -> Sensor exists via BFS
    visited: Set[str] = set()
    queue: deque[str] = deque([source_id])
    visited.add(source_id)
    while queue:
        current = queue.popleft()
        for neighbor in adj.get(current, []):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    if sensor_id not in visited:
        raise GraphCompilationError(
            f"No directed path from source '{source_id}' to "
            f"sensor '{sensor_id}'"
        )

    if noise_id not in visited:
        raise GraphCompilationError(
            f"No directed path from source '{source_id}' to "
            f"noise '{noise_id}'"
        )

    # ---------------------------------------------------------------
    # Carrier-transition enforcement (R2)
    # ---------------------------------------------------------------
    carrier_transitions = spec.metadata.get("carrier_transitions", None)
    if carrier_transitions:
        # Build set of physics_subroles present in element nodes
        element_subroles: set = set()
        for node in spec.nodes:
            if node.node_id in elements:
                # Check explicit physics_subrole on node
                if node.physics_subrole is not None:
                    element_subroles.add(node.physics_subrole.value if hasattr(node.physics_subrole, 'value') else str(node.physics_subrole))
                # Check primitive class attribute
                prim_cls = PRIMITIVE_REGISTRY.get(node.primitive_id)
                if prim_cls is not None:
                    sr = getattr(prim_cls, "_physics_subrole", None)
                    if sr is not None:
                        element_subroles.add(sr)
        transition_subroles = {"interaction", "transduction"}
        if not element_subroles & transition_subroles:
            raise GraphCompilationError(
                f"carrier_transitions declared {carrier_transitions} but no "
                f"element node has physics_subrole in {transition_subroles}. "
                f"Found subroles: {element_subroles}"
            )

    logger.debug(
        f"Canonical chain validated: {source_id} -> "
        f"[{', '.join(elements)}] -> {sensor_id} -> {noise_id}"
    )
