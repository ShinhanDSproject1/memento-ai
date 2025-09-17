from pydantic import BaseModel
from typing import List
class RecommendationRequest(BaseModel):
    member_seq:int
    queries: List[str]