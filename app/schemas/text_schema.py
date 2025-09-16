from pydantic import BaseModel
from typing import List
class TextRequest(BaseModel):
    queries: List[str]