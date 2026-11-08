from typing import Any, Dict, List, Optional, Set, Tuple
import networkx as nx

from models import Entity, GraphChunk, Relation


class KnowledgeGraphStore:
    """
    In-memory Knowledge Graph store powered by NetworkX.
    Maintains:
    - (:Entity)-[:RELATION {type, chunk_id}]->(:Entity)
    - (:Chunk)-[:MENTIONS]->(:Entity)
    Provides BFS multi-hop expansion and chunk attribution.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.chunk_store: Dict[str, GraphChunk] = {}
        self.entity_names: Set[str] = set()

    def add_chunk(self, chunk: GraphChunk):
        """Index a document chunk and its MENTIONS edges."""
        self.chunk_store[chunk.chunk_id] = chunk
        chunk_node_id = f"chunk:{chunk.chunk_id}"
        self.graph.add_node(
            chunk_node_id,
            node_type="CHUNK",
            chunk_id=chunk.chunk_id,
            title=chunk.title,
            text=chunk.text,
        )

    def add_entity(self, entity: Entity):
        """Add an entity node."""
        self.entity_names.add(entity.name)
        self.graph.add_node(
            entity.name,
            node_type="ENTITY",
            entity_type=entity.entity_type.value if hasattr(entity.entity_type, "value") else str(entity.entity_type),
            aliases=entity.aliases,
        )

    def add_relation(self, relation: Relation):
        """
        Add a typed directed relation between two entities,
        recording chunk_id as provenance. Also adds bidirectional traversal metadata.
        """
        self.entity_names.add(relation.source)
        self.entity_names.add(relation.target)

        # Ensure entity nodes exist
        if not self.graph.has_node(relation.source):
            self.graph.add_node(relation.source, node_type="ENTITY")
        if not self.graph.has_node(relation.target):
            self.graph.add_node(relation.target, node_type="ENTITY")

        # Add forward edge
        self.graph.add_edge(
            relation.source,
            relation.target,
            relation_type=relation.relation_type,
            chunk_id=relation.chunk_id,
        )

        # Connect originating chunk to both entities with MENTIONS
        chunk_node_id = f"chunk:{relation.chunk_id}"
        if not self.graph.has_node(chunk_node_id):
            self.graph.add_node(chunk_node_id, node_type="CHUNK", chunk_id=relation.chunk_id)

        self.graph.add_edge(chunk_node_id, relation.source, relation_type="MENTIONS", chunk_id=relation.chunk_id)
        self.graph.add_edge(chunk_node_id, relation.target, relation_type="MENTIONS", chunk_id=relation.chunk_id)

    def link_entities_from_text(self, text: str) -> List[str]:
        """
        Extract question seed entities by matching known entity names and aliases
        against the input text (case-insensitive word boundary match).
        """
        text_lower = text.lower()
        matched = []
        # Sort by length descending to match longest phrases first (e.g. "Christopher Nolan" before "Nolan")
        sorted_entities = sorted(self.entity_names, key=len, reverse=True)

        for entity in sorted_entities:
            # Check direct match
            if entity.lower() in text_lower:
                matched.append(entity)
                continue

            # Check aliases
            node_data = self.graph.nodes.get(entity, {})
            aliases = node_data.get("aliases", [])
            for alias in aliases:
                if alias.lower() in text_lower:
                    matched.append(entity)
                    break

        return list(dict.fromkeys(matched))

    def expand_multihop(self, seed_entities: List[str], max_hops: int = 2) -> Tuple[Set[str], Set[str], List[Dict[str, Any]]]:
        """
        Traverse graph from seed entities up to max_hops (undirected across entities).
        Returns:
          - visited_entities: Set[str]
          - supporting_chunk_ids: Set[str]
          - traversed_triples: List[Dict[str, Any]]
        """
        if not seed_entities:
            return set(), set(), []

        visited_entities: Set[str] = set()
        supporting_chunk_ids: Set[str] = set()
        traversed_triples: List[Dict[str, Any]] = []

        # Convert to undirected view for graph traversal across relationship edges
        frontier = set(seed_entities)
        visited_entities.update(seed_entities)

        for hop in range(max_hops):
            next_frontier = set()
            for u in frontier:
                if not self.graph.has_node(u):
                    continue

                # Outgoing edges
                for _, v, key, data in self.graph.out_edges(u, keys=True, data=True):
                    rel_type = data.get("relation_type", "")
                    chunk_id = data.get("chunk_id")
                    if chunk_id:
                        supporting_chunk_ids.add(chunk_id)

                    if not v.startswith("chunk:") and v not in visited_entities:
                        visited_entities.add(v)
                        next_frontier.add(v)
                        traversed_triples.append({
                            "source": u,
                            "relation": rel_type,
                            "target": v,
                            "chunk_id": chunk_id,
                            "hop": hop + 1,
                        })

                # Incoming edges
                for v, _, key, data in self.graph.in_edges(u, keys=True, data=True):
                    rel_type = data.get("relation_type", "")
                    chunk_id = data.get("chunk_id")
                    if chunk_id:
                        supporting_chunk_ids.add(chunk_id)

                    if not v.startswith("chunk:") and v not in visited_entities:
                        visited_entities.add(v)
                        next_frontier.add(v)
                        traversed_triples.append({
                            "source": v,
                            "relation": rel_type,
                            "target": u,
                            "chunk_id": chunk_id,
                            "hop": hop + 1,
                        })

            frontier = next_frontier

        # Also pull chunks that directly MENTION any visited entity
        for ent in visited_entities:
            for v, _, data in self.graph.in_edges(ent, data=True):
                if v.startswith("chunk:"):
                    cid = data.get("chunk_id")
                    if cid:
                        supporting_chunk_ids.add(cid)

        return visited_entities, supporting_chunk_ids, traversed_triples

    def format_triples(self, triples: List[Dict[str, Any]]) -> List[str]:
        """Convert traversed triple dicts to human-readable strings."""
        formatted = []
        seen = set()
        for t in triples:
            rep = f"({t['source']}) -[{t['relation']}]-> ({t['target']})"
            if rep not in seen:
                seen.add(rep)
                formatted.append(rep)
        return formatted

    def get_stats(self) -> Dict[str, Any]:
        """Return graph size and connectivity metrics."""
        entity_nodes = [n for n in self.graph.nodes if not n.startswith("chunk:")]
        chunk_nodes = [n for n in self.graph.nodes if n.startswith("chunk:")]
        relation_edges = [
            (u, v) for u, v, d in self.graph.edges(data=True)
            if not u.startswith("chunk:") and not v.startswith("chunk:")
        ]
        return {
            "num_entities": len(entity_nodes),
            "num_chunks": len(chunk_nodes),
            "num_relations": len(relation_edges),
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
        }
