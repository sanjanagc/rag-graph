import pytest
from extractor import GraphExtractor
from graph_store import KnowledgeGraphStore
from models import Entity, EntityType, GraphChunk, Relation
from pipeline import HybridGraphRAGPipeline
from resolver import EntityResolver


@pytest.fixture(scope="module")
def pipeline():
    return HybridGraphRAGPipeline()


# =========================================================================
# TEST 1: Entity & Relation Extraction
# =========================================================================
def test_entity_relation_extraction():
    extractor = GraphExtractor(use_llm_if_available=False)
    chunk = GraphChunk(
        chunk_id="chunk_test_01",
        title="Test Inception",
        text="Inception is a 2010 science fiction heist film directed by Christopher Nolan, starring Leonardo DiCaprio as Dom Cobb.",
        entities=["Inception", "Christopher Nolan", "Leonardo DiCaprio"],
    )
    graph = extractor.extract(chunk)
    assert len(graph.entities) >= 2
    assert any(e.name == "Christopher Nolan" for e in graph.entities)

    # Check relation binding chunk_id
    assert len(graph.relations) >= 1
    for r in graph.relations:
        assert r.chunk_id == "chunk_test_01"


# =========================================================================
# TEST 2: Entity Resolution (Deduplication)
# =========================================================================
def test_entity_resolution():
    resolver = EntityResolver()
    extractor = GraphExtractor(use_llm_if_available=False)

    chunk1 = GraphChunk(
        chunk_id="c1",
        text="Inception was directed by Christopher Nolan.",
        entities=["Christopher Nolan"],
    )
    chunk2 = GraphChunk(
        chunk_id="c2",
        text="C. Nolan was born in London.",
        entities=["C. Nolan"],
    )
    g1 = extractor.extract(chunk1)
    g2 = extractor.extract(chunk2)

    resolved_entities, resolved_relations = resolver.resolve_entities([g1, g2])
    entity_names = [e.name for e in resolved_entities]

    # Verify that "C. Nolan" was merged into canonical "Christopher Nolan"
    assert "Christopher Nolan" in entity_names
    assert "C. Nolan" not in entity_names


# =========================================================================
# TEST 3: Multi-Hop Traversal (2 Hops)
# =========================================================================
def test_multihop_graph_expansion(pipeline):
    # From "Oppenheimer", 1 hop -> Christopher Nolan, 2 hops -> Westminster, London & UCL
    visited_nodes, chunk_ids, triples = pipeline.graph_store.expand_multihop(
        seed_entities=["Oppenheimer"], max_hops=2
    )
    assert "Christopher Nolan" in visited_nodes
    assert any("London" in n or "Westminster" in n for n in visited_nodes)
    assert "chunk_oppenheimer" in chunk_ids
    assert "chunk_nolan" in chunk_ids


# =========================================================================
# TEST 4: Bridge Question Evidence Gap (Vector vs GraphRAG)
# =========================================================================
def test_bridge_question_evidence_gap(pipeline):
    query = "Where was the director of Oppenheimer born?"

    # 1. Vector only: query has "director", "born", "Oppenheimer"
    # Typically misses chunk_nolan or ranks it poorly because chunk_nolan never mentions "Oppenheimer"
    vec_chunks = pipeline.retrieve_vector_only(query, top_k=2)
    vec_ids = [c.chunk_id for c in vec_chunks]

    # 2. Hybrid GraphRAG: traverses Oppenheimer -> Nolan -> birthplace
    hybrid_chunks, triples = pipeline.retrieve_hybrid(query, top_k=4)
    hybrid_ids = [c.chunk_id for c in hybrid_chunks]

    # Hybrid GraphRAG MUST retrieve both supporting chunks
    assert "chunk_oppenheimer" in hybrid_ids, "GraphRAG missed chunk_oppenheimer!"
    assert "chunk_nolan" in hybrid_ids, "GraphRAG missed chunk_nolan!"

    # Verify triples explain the connection
    assert any("DIRECTED" in t for t in triples)
    assert any("BORN_IN" in t for t in triples)


# =========================================================================
# TEST 5: Single-Hop Control Question Stability
# =========================================================================
def test_single_hop_retrieval(pipeline):
    query = "When was Apex Technologies founded?"
    hybrid_chunks, _ = pipeline.retrieve_hybrid(query, top_k=2)
    chunk_ids = [c.chunk_id for c in hybrid_chunks]

    assert "chunk_apex_tech" in chunk_ids
    answer = pipeline.generate_answer(query, hybrid_chunks)
    assert "2019" in answer
