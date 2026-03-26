"""pwm_core.core.evidence_graph

Scientific evidence graph -- typed relationships between PWM objects.

The evidence graph enables:
- Provenance tracing: Certificate -> spec -> method -> dataset -> instrument
- Impact analysis: when a DatasetCard is updated, find affected RunBundles
- Reproducibility verification: ClaimCard -> independent RunBundle
- Trust propagation: Certified Certificate strengthens referenced cards

Stored as a lightweight in-memory graph with JSON serialization.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import datetime


class EdgeType(str, Enum):
    """Typed relationship between two PWM objects."""
    references = "references"       # ClaimCard -> SpecCard
    compiles_to = "compiles_to"     # SpecCard -> OperatorGraph
    executes = "executes"           # OperatorGraph -> RunBundle
    solves = "solves"               # MethodCard -> RunBundle
    uses_dataset = "uses_dataset"   # RunBundle -> DatasetCard
    acquired_by = "acquired_by"     # DatasetCard -> InstrumentCard
    judged_by = "judged_by"         # RunBundle -> Certificate
    associated_with = "associated_with"  # EventCard -> ClaimCard


class NodeType(str, Enum):
    claim_card = "ClaimCard"
    spec_card = "SpecCard"
    method_card = "MethodCard"
    dataset_card = "DatasetCard"
    instrument_card = "InstrumentCard"
    event_card = "EventCard"
    operator_graph = "OperatorGraph"
    run_bundle = "RunBundle"
    certificate = "Certificate"


@dataclass
class GraphNode:
    node_id: str
    node_type: NodeType
    version: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GraphEdge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc).isoformat()
    )


class EvidenceGraph:
    """In-memory scientific evidence graph with JSON serialization."""

    def __init__(self) -> None:
        self._nodes: Dict[str, GraphNode] = {}
        self._edges: List[GraphEdge] = []
        self._adjacency: Dict[str, List[GraphEdge]] = {}  # source_id -> edges
        self._reverse: Dict[str, List[GraphEdge]] = {}    # target_id -> edges

    # -- mutation -------------------------------------------------------

    def add_node(
        self,
        node_id: str,
        node_type: NodeType,
        version: Optional[str] = None,
        **metadata: Any,
    ) -> GraphNode:
        """Insert or replace a node."""
        node = GraphNode(
            node_id=node_id,
            node_type=node_type,
            version=version,
            metadata=metadata,
        )
        self._nodes[node_id] = node
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        **metadata: Any,
    ) -> GraphEdge:
        """Append a directed edge.  Nodes need not exist yet."""
        edge = GraphEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            metadata=metadata,
        )
        self._edges.append(edge)
        self._adjacency.setdefault(source_id, []).append(edge)
        self._reverse.setdefault(target_id, []).append(edge)
        return edge

    # -- queries --------------------------------------------------------

    def get_node(self, node_id: str) -> Optional[GraphNode]:
        return self._nodes.get(node_id)

    def neighbors(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[str]:
        """Return IDs reachable via one outgoing edge."""
        edges = self._adjacency.get(node_id, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return [e.target_id for e in edges]

    def predecessors(
        self, node_id: str, edge_type: Optional[EdgeType] = None
    ) -> List[str]:
        """Return IDs with an incoming edge to *node_id*."""
        edges = self._reverse.get(node_id, [])
        if edge_type is not None:
            edges = [e for e in edges if e.edge_type == edge_type]
        return [e.source_id for e in edges]

    def trace_provenance(self, certificate_id: str) -> Dict[str, List[str]]:
        """Walk backward from a Certificate to find all contributing objects."""
        result: Dict[str, List[str]] = {}
        visited: Set[str] = set()
        queue = [certificate_id]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            node = self._nodes.get(nid)
            if node is not None:
                result.setdefault(node.node_type.value, []).append(nid)
            for pred_id in self.predecessors(nid):
                if pred_id not in visited:
                    queue.append(pred_id)
        return result

    def impact_analysis(self, node_id: str) -> Dict[str, List[str]]:
        """Walk forward from a node to find all affected objects."""
        result: Dict[str, List[str]] = {}
        visited: Set[str] = set()
        queue = [node_id]
        while queue:
            nid = queue.pop(0)
            if nid in visited:
                continue
            visited.add(nid)
            node = self._nodes.get(nid)
            if node is not None:
                result.setdefault(node.node_type.value, []).append(nid)
            for succ_id in self.neighbors(nid):
                if succ_id not in visited:
                    queue.append(succ_id)
        return result

    def find_by_type(self, node_type: NodeType) -> List[GraphNode]:
        """Return all nodes of a given type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    # -- properties -----------------------------------------------------

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    # -- serialization --------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        import dataclasses
        return {
            "nodes": {
                nid: dataclasses.asdict(n) for nid, n in self._nodes.items()
            },
            "edges": [dataclasses.asdict(e) for e in self._edges],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvidenceGraph":
        g = cls()
        for nid, nd in data.get("nodes", {}).items():
            g.add_node(
                nid,
                NodeType(nd["node_type"]),
                nd.get("version"),
                **nd.get("metadata", {}),
            )
        for ed in data.get("edges", []):
            g.add_edge(
                ed["source_id"],
                ed["target_id"],
                EdgeType(ed["edge_type"]),
                **ed.get("metadata", {}),
            )
        return g

    @classmethod
    def from_json(cls, text: str) -> "EvidenceGraph":
        return cls.from_dict(json.loads(text))

    # -- certificate-based construction ---------------------------------

    @classmethod
    def build_from_certificate(cls, cert_path: str) -> "EvidenceGraph":
        """Build an evidence graph from a certificate.json file.

        Creates nodes for the Certificate, its RunBundle, and referenced
        spec/method/dataset, then links them with typed edges.
        """
        from pathlib import Path

        p = Path(cert_path)
        cert = json.loads(p.read_text())
        bundle_dir = p.parent

        g = cls()

        # Certificate node
        cert_id = cert.get("run_id", "unknown_cert")
        g.add_node(
            cert_id + "_cert",
            NodeType.certificate,
            trust_tier=cert.get("trust_tier"),
            judge_version=cert.get("judge_version"),
        )

        # RunBundle node
        g.add_node(
            cert_id,
            NodeType.run_bundle,
            version=cert.get("kernel_version"),
        )
        g.add_edge(cert_id, cert_id + "_cert", EdgeType.judged_by)

        # Spec node (from spec_id)
        spec_id = cert.get("spec_id", "")
        if spec_id:
            g.add_node(spec_id, NodeType.spec_card)
            g.add_edge(spec_id, cert_id, EdgeType.compiles_to)

        # Read manifest for more detail
        manifest_path = bundle_dir / "runbundle_manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            prov = manifest.get("provenance", {})

            # Method node
            solver = prov.get("solver", "")
            if solver:
                method_id = f"method_{prov.get('modality', '')}_{solver}"
                g.add_node(method_id, NodeType.method_card, solver=solver)
                g.add_edge(method_id, cert_id, EdgeType.solves)

            # Dataset node
            modality = prov.get("modality", "")
            if modality:
                dataset_id = f"dataset_{modality}_public"
                g.add_node(dataset_id, NodeType.dataset_card, modality=modality)
                g.add_edge(cert_id, dataset_id, EdgeType.uses_dataset)

        return g

    # -- convenience queries --------------------------------------------

    def impact_analysis_from_certificate(
        self, cert_path: str
    ) -> Dict[str, List[str]]:
        """Build a graph from a certificate and return its full impact map.

        Combines ``build_from_certificate`` with ``impact_analysis`` so
        callers can go straight from a certificate file to a dict of
        affected node-type -> node-ids.
        """
        g = self.build_from_certificate(cert_path)
        # Find the certificate node and trace forward from it
        cert_nodes = g.find_by_type(NodeType.certificate)
        if not cert_nodes:
            return {}
        return g.impact_analysis(cert_nodes[0].node_id)

    def find_affected_by_dataset(self, dataset_id: str) -> List[str]:
        """Find all Certificates affected by changes to a dataset.

        Because RunBundles point *to* datasets via ``uses_dataset``, a
        forward walk from the dataset node alone would not reach the
        RunBundles.  This method first collects all predecessors of the
        dataset (i.e. RunBundles that use it), then walks forward from
        each to collect reachable Certificate nodes.
        """
        certificate_ids: List[str] = []
        seen: Set[str] = set()

        # Predecessors of the dataset are RunBundles (or other nodes)
        # that reference it via uses_dataset edges.
        for pred_id in self.predecessors(dataset_id):
            affected = self.impact_analysis(pred_id)
            for cid in affected.get("Certificate", []):
                if cid not in seen:
                    seen.add(cid)
                    certificate_ids.append(cid)

        # Also check direct forward impact (if dataset has outgoing edges)
        direct = self.impact_analysis(dataset_id)
        for cid in direct.get("Certificate", []):
            if cid not in seen:
                seen.add(cid)
                certificate_ids.append(cid)

        return certificate_ids
