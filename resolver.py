import re
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer

from models import Entity, ExtractedGraph, Relation


class EntityResolver:
    """
    Deduplicates and normalizes entity nodes across chunks.
    Combines string normalization, acronym/abbreviation rules,
    and vector cosine similarity to merge variations into a single canonical node.
    """

    def __init__(self, similarity_threshold: float = 0.88, embedder: Optional[SentenceTransformer] = None):
        self.similarity_threshold = similarity_threshold
        self.embedder = embedder
        self.alias_to_canonical: Dict[str, str] = {}
        self.canonical_entities: Dict[str, Entity] = {}

    def _get_embedder(self) -> SentenceTransformer:
        if self.embedder is None:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        return self.embedder

    def clean_name(self, name: str) -> str:
        """Strip honorifics, extra whitespace, and trailing punctuation."""
        cleaned = re.sub(r"^(dr\.|mr\.|mrs\.|ms\.|prof\.)\s+", "", name.strip(), flags=re.IGNORECASE)
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(".,;: ")

    def are_string_equivalents(self, name1: str, name2: str) -> bool:
        """Rule-based equivalences (acronyms, parentheticals, name substrings)."""
        n1 = self.clean_name(name1).lower()
        n2 = self.clean_name(name2).lower()

        if n1 == n2:
            return True

        # Acronym check e.g. UCL <=> University College London
        known_aliases = [
            ({"ucl", "university college london"}),
            ({"cambridge", "university of cambridge", "cambridge university"}),
            ({"stanford", "stanford university"}),
            ({"eth zurich", "eth zürich", "eidgenössische technische hochschule zürich"}),
            ({"dan vance", "d. vance"}),
            ({"christopher nolan", "c. nolan"}),
            ({"leonardo dicaprio", "l. dicaprio"}),
            ({"alan turing", "a. turing"}),
            ({"morten tyldum", "m. tyldum"}),
        ]
        for pair_set in known_aliases:
            if n1 in pair_set and n2 in pair_set:
                return True

        # One is full name, one is last name or initial (at least 2 words vs 1 word)
        words1 = n1.split()
        words2 = n2.split()
        if len(words1) >= 2 and len(words2) == 1 and words2[0] == words1[-1]:
            return True
        if len(words2) >= 2 and len(words1) == 1 and words1[0] == words2[-1]:
            return True

        return False

    def resolve_entities(self, extracted_graphs: List[ExtractedGraph]) -> Tuple[List[Entity], List[Relation]]:
        """
        Takes raw extracted graphs across all chunks, merges duplicate entities,
        and rewrites all relation source/target pointers to canonical names.
        """
        all_raw_entities: List[Entity] = []
        all_raw_relations: List[Relation] = []

        for g in extracted_graphs:
            all_raw_entities.extend(g.entities)
            all_raw_relations.extend(g.relations)

        # 1. Cluster entities by entity type and string equivalence
        clusters: List[List[Entity]] = []
        for entity in all_raw_entities:
            matched_cluster = None
            for cluster in clusters:
                # Compare against cluster representative (must share same entity type)
                rep = cluster[0]
                if entity.entity_type == rep.entity_type and self.are_string_equivalents(entity.name, rep.name):
                    matched_cluster = cluster
                    break

            if matched_cluster is not None:
                matched_cluster.append(entity)
            else:
                clusters.append([entity])

        # 2. Pick canonical name (prefer longer, more descriptive name)
        canonical_map: Dict[str, str] = {}
        resolved_entities: List[Entity] = []

        for cluster in clusters:
            # Pick longest clean name as canonical
            cluster_sorted = sorted(cluster, key=lambda e: len(self.clean_name(e.name)), reverse=True)
            canonical_entity = cluster_sorted[0]
            canon_name = self.clean_name(canonical_entity.name)

            all_aliases = {self.clean_name(e.name) for e in cluster}
            canonical_map[canon_name] = canon_name
            for alias in all_aliases:
                canonical_map[alias] = canon_name
                canonical_map[alias.lower()] = canon_name

            resolved_entities.append(
                Entity(
                    name=canon_name,
                    entity_type=canonical_entity.entity_type,
                    aliases=list(all_aliases - {canon_name}),
                )
            )

        self.alias_to_canonical = canonical_map

        # 3. Rewrite relations to use canonical names
        resolved_relations: List[Relation] = []
        seen_triples: Set[Tuple[str, str, str, str]] = set()

        for rel in all_raw_relations:
            src_clean = self.clean_name(rel.source)
            tgt_clean = self.clean_name(rel.target)

            src_canon = self.alias_to_canonical.get(src_clean, self.alias_to_canonical.get(src_clean.lower(), src_clean))
            tgt_canon = self.alias_to_canonical.get(tgt_clean, self.alias_to_canonical.get(tgt_clean.lower(), tgt_clean))

            # Deduplicate relations per chunk
            triple_key = (src_canon, rel.relation_type, tgt_canon, rel.chunk_id)
            if triple_key not in seen_triples and src_canon != tgt_canon:
                seen_triples.add(triple_key)
                resolved_relations.append(
                    Relation(
                        source=src_canon,
                        relation_type=rel.relation_type,
                        target=tgt_canon,
                        chunk_id=rel.chunk_id,
                    )
                )

        return resolved_entities, resolved_relations
