"""pwm_core.graph.compiler
==========================

GraphCompiler: validate -> bind -> plan -> export.

Compilation pipeline
--------------------
1. **Validate:**  DAG check, all primitive_ids exist, no duplicate node_ids.
2. **Bind:**      Instantiate primitives with parameters theta.
3. **Plan fwd:**  Topological sort -> sequential execution plan.
4. **Plan adj:**  Reverse topological order (for linear graphs).
5. **Export:**    Return ``GraphOperator`` ready for forward/adjoint/serialize.
"""

from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Set, Tuple

from pwm_core.graph.graph_operator import GraphOperator
from pwm_core.graph.graph_spec import OperatorGraphSpec
from pwm_core.graph.ir_types import NodeTags
from pwm_core.graph.primitives import PRIMITIVE_REGISTRY, BasePrimitive, get_primitive

logger = logging.getLogger(__name__)


class GraphCompilationError(Exception):
    """Raised when an OperatorGraphSpec cannot be compiled."""


class GraphCompiler:
    """Compile an OperatorGraphSpec into a GraphOperator.

    Usage
    -----
    >>> compiler = GraphCompiler()
    >>> op = compiler.compile(spec)
    >>> y = op.forward(x)
    """

    def compile(
        self,
        spec: OperatorGraphSpec,
        x_shape: Optional[Tuple[int, ...]] = None,
        y_shape: Optional[Tuple[int, ...]] = None,
    ) -> GraphOperator:
        """Full compilation pipeline.

        Parameters
        ----------
        spec : OperatorGraphSpec
            The graph specification to compile.
        x_shape : tuple, optional
            Input shape override (default: from metadata or (64, 64)).
        y_shape : tuple, optional
            Output shape override (default: from metadata or same as x_shape).

        Returns
        -------
        GraphOperator
            Compiled, ready-to-run operator graph.

        Raises
        ------
        GraphCompilationError
            If validation fails (cycle, missing primitive, etc.).
        """
        # Step 1: Validate
        self._validate_dag(spec)
        self._validate_primitive_ids(spec)

        # Step 1b: Canonical chain validation (opt-in via metadata flag)
        if spec.metadata.get("canonical_chain", False):
            from pwm_core.graph.canonical import validate_canonical_chain
            validate_canonical_chain(spec)

        # Step 2: Topological sort
        topo_order = self._topological_sort(spec)

        # Step 3: Bind — instantiate primitives with params
        node_map: Dict[str, BasePrimitive] = {}
        for node in spec.nodes:
            prim = get_primitive(node.primitive_id, params=node.params)
            node_map[node.node_id] = prim

        # Step 3b: Build edge_map and validate multi-input constraints
        edge_map: Dict[str, List[str]] = {}
        for edge in spec.edges:
            if edge.target not in edge_map:
                edge_map[edge.target] = []
            edge_map[edge.target].append(edge.source)

        # Validate: for multi-input nodes, incoming edge count must match _n_inputs
        for node_id, prim in node_map.items():
            n_inputs = getattr(prim, '_n_inputs', 1)
            n_incoming = len(edge_map.get(node_id, []))
            # Only validate for multi-input primitives
            if n_inputs > 1:
                if n_incoming != n_inputs:
                    raise GraphCompilationError(
                        f"Node '{node_id}' has {n_incoming} incoming edges but "
                        f"primitive '{prim.primitive_id}' expects {n_inputs} inputs"
                    )

        # Step 4: Build forward and adjoint plans
        forward_plan: List[Tuple[str, BasePrimitive]] = [
            (nid, node_map[nid]) for nid in topo_order
        ]
        adjoint_plan: List[Tuple[str, BasePrimitive]] = list(
            reversed(forward_plan)
        )

        # Check linearity
        all_linear = all(p.is_linear for _, p in forward_plan)

        # Determine shapes
        _x_shape = x_shape or tuple(
            spec.metadata.get("x_shape", [64, 64])
        )
        _y_shape = y_shape or tuple(
            spec.metadata.get("y_shape", list(_x_shape))
        )

        # Collect learnable params
        learnable_params: Dict[str, List[str]] = {}
        for node in spec.nodes:
            if node.learnable:
                learnable_params[node.node_id] = list(node.learnable)

        # Derive NodeTags from primitive attributes
        node_tags: Dict[str, NodeTags] = {}
        for node_id, prim in node_map.items():
            _tier_str = getattr(prim, '_physics_tier', None)
            _physics_tier = None
            if _tier_str is not None:
                from pwm_core.graph.ir_types import PhysicsTier as _PT
                try:
                    _physics_tier = _PT(_tier_str)
                except ValueError:
                    pass

            _subrole_str = getattr(prim, '_physics_subrole', None)
            _physics_subrole = None
            if _subrole_str is not None:
                from pwm_core.graph.ir_types import PhysicsSubrole as _PSR
                try:
                    _physics_subrole = _PSR(_subrole_str)
                except ValueError:
                    pass

            node_tags[node_id] = NodeTags(
                is_linear=prim.is_linear,
                is_stochastic=prim.is_stochastic,
                is_differentiable=prim.is_differentiable,
                is_stateful=prim.is_stateful,
                physics_tier=_physics_tier,
                physics_subrole=_physics_subrole,
            )

        # Collect edges
        edges_list = [(e.source, e.target) for e in spec.edges]

        graph_op = GraphOperator(
            graph_id=spec.graph_id,
            forward_plan=forward_plan,
            adjoint_plan=adjoint_plan,
            node_map=node_map,
            all_linear=all_linear,
            x_shape=_x_shape,
            y_shape=_y_shape,
            metadata=spec.metadata,
            edges=edges_list,
            learnable_params=learnable_params,
            node_tags=node_tags,
            edge_map=edge_map,
            spec=spec,
        )

        logger.info(
            f"Compiled graph '{spec.graph_id}': "
            f"{len(forward_plan)} nodes, {len(edges_list)} edges, "
            f"all_linear={all_linear}"
        )

        return graph_op

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_dag(self, spec: OperatorGraphSpec) -> None:
        """Verify that the graph is a DAG (no cycles)."""
        node_ids = {n.node_id for n in spec.nodes}

        # Build adjacency list
        adj: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in spec.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        # Kahn's algorithm for cycle detection
        queue: deque[str] = deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        visited = 0

        while queue:
            node = queue.popleft()
            visited += 1
            for neighbour in adj[node]:
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)

        if visited != len(node_ids):
            raise GraphCompilationError(
                f"Graph '{spec.graph_id}' contains a cycle. "
                f"Visited {visited}/{len(node_ids)} nodes."
            )

    def _validate_primitive_ids(self, spec: OperatorGraphSpec) -> None:
        """Ensure every node references a known primitive_id."""
        for node in spec.nodes:
            if node.primitive_id not in PRIMITIVE_REGISTRY:
                raise GraphCompilationError(
                    f"Node '{node.node_id}' references unknown "
                    f"primitive_id '{node.primitive_id}'. "
                    f"Available: {sorted(PRIMITIVE_REGISTRY.keys())}"
                )

    def _topological_sort(self, spec: OperatorGraphSpec) -> List[str]:
        """Return node_ids in topological order (Kahn's algorithm).

        Nodes with no incoming edges come first (sources).
        If there are multiple valid orderings, we prefer the order
        in which nodes appear in the spec to ensure determinism.
        """
        node_ids = [n.node_id for n in spec.nodes]
        node_set = set(node_ids)

        # Build adjacency + in-degree
        adj: Dict[str, List[str]] = defaultdict(list)
        in_degree: Dict[str, int] = {nid: 0 for nid in node_ids}

        for edge in spec.edges:
            adj[edge.source].append(edge.target)
            in_degree[edge.target] = in_degree.get(edge.target, 0) + 1

        # Use a priority queue where priority = original index
        # to ensure deterministic ordering.
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        queue: List[str] = sorted(
            [nid for nid, deg in in_degree.items() if deg == 0],
            key=lambda nid: idx_map[nid],
        )

        result: List[str] = []
        while queue:
            node = queue.pop(0)  # take smallest index first
            result.append(node)
            for neighbour in sorted(adj[node], key=lambda n: idx_map[n]):
                in_degree[neighbour] -= 1
                if in_degree[neighbour] == 0:
                    queue.append(neighbour)
            queue.sort(key=lambda nid: idx_map[nid])

        if len(result) != len(node_ids):
            # Should have been caught by _validate_dag, but just in case
            raise GraphCompilationError(
                f"Topological sort failed: {len(result)}/{len(node_ids)} "
                "nodes sorted (possible cycle)."
            )

        return result

    # ------------------------------------------------------------------
    # Utility: compile from dict (for YAML loading)
    # ------------------------------------------------------------------

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> OperatorGraphSpec:
        """Parse a dict (e.g. from YAML) into an OperatorGraphSpec.

        This handles the common pattern where graph templates are stored
        in YAML as::

            graph_id: cassi_sd_graph_v1
            nodes:
              - node_id: modulate
                primitive_id: coded_mask
                params: {seed: 42, H: 64, W: 64}
            edges:
              - source: modulate
                target: disperse

        Returns
        -------
        OperatorGraphSpec
            Validated Pydantic model.
        """
        return OperatorGraphSpec.model_validate(data)
