from pydantic import BaseModel

class TextRequest(BaseModel):
    queries: list[str]