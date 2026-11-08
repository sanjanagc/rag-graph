from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field


class EntityType(str, Enum):
    PERSON = "PERSON"
    WORK = "WORK"          # Film, Book, Album, Product
    LOCATION = "LOCATION"  # City, Country, Building
    ORGANIZATION = "ORGANIZATION"  # University, Company, Lab
    EVENT = "EVENT"
    CONCEPT = "CONCEPT"


class RelationType(str, Enum):
    DIRECTED = "DIRECTED"
    BORN_IN = "BORN_IN"
    FOUNDED = "FOUNDED"
    LOCATED_IN = "LOCATED_IN"
    GRADUATED_FROM = "GRADUATED_FROM"
    AWARDED = "AWARDED"
    STARRED_IN = "STARRED_IN"
    PRODUCED = "PRODUCED"
    MEMBER_OF = "MEMBER_OF"
    INVENTED = "INVENTED"
    ASSOCIATED_WITH = "ASSOCIATED_WITH"


class Entity(BaseModel):
    name: str
    entity_type: EntityType
    aliases: List[str] = Field(default_factory=list)


class Relation(BaseModel):
    source: str
    relation_type: str
    target: str
    chunk_id: str


class ExtractedGraph(BaseModel):
    entities: List[Entity] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)


class GraphChunk(BaseModel):
    chunk_id: str
    title: str = ""
    text: str
    entities: List[str] = Field(default_factory=list)


class MultiHopQuestion(BaseModel):
    question_id: str
    question: str
    expected_answer: str
    supporting_chunk_ids: List[str]
    is_multihop: bool = True


class RetrievedChunk(BaseModel):
    chunk_id: str
    title: str
    text: str
    score: float
    retrieval_path: str = "hybrid"  # "vector", "graph", "hybrid"


class GraphRAGResponse(BaseModel):
    question: str
    answer: str
    retrieved_chunks: List[RetrievedChunk]
    supporting_triples: List[str]
    retrieval_latency_ms: float
