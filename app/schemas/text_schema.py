from pydantic import BaseModel
from typing import List
class RecommendationRequest(BaseModel):
    member_seq:str
    queries: List[str]