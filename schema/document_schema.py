from pydantic import BaseModel, Field, field_validator # type: ignore
from typing import List

# CHUNK MODEL

class ChunkSummary(BaseModel):
    chunk_id: int
    summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    key_risks_action_items: List[str] = Field(default_factory=list)

# SECTION MODEL

class SectionSummary(BaseModel):
    section_id: int
    section_summary: str = ""
    section_key_points: List[str] = Field(default_factory=list)
    section_risks_action_items: List[str] = Field(default_factory=list)
    covered_chunk_ids: List[int] = Field(default_factory=list)

    @field_validator("covered_chunk_ids")
    @classmethod
    def validate_ids(cls, v):
        return [int(i) for i in v if isinstance(i, (int, str))]

# FINAL DOCUMENT MODEL

class DocumentSummary(BaseModel):
    tldr: str = ""
    executive_summary: str = ""
    key_points: List[str] = Field(default_factory=list)
    risks_action_items: List[str] = Field(default_factory=list)
    sections: List[SectionSummary] = Field(default_factory=list)
    chunk_summaries: List[ChunkSummary] = Field(default_factory=list)
    coverage_score: float = Field(default=0.0, ge=0.0, le=100.0)
    coverage_flag: bool = False
    missing_section_flag: bool = False
    meaning_coverage_score: float = Field(default=0.0, ge=0.0, le=100.0)


class FinalOutput(BaseModel):
    document_summary: DocumentSummary
