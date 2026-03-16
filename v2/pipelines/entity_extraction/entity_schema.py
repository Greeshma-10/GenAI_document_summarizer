from pydantic import BaseModel
from typing import List


class Entities(BaseModel):
    models: List[str]
    datasets: List[str]
    metrics: List[str]
    organizations: List[str]
    tasks: List[str]
    key_concepts: List[str]