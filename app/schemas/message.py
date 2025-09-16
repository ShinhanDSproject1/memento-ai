from pydantic import BaseModel

class Message(BaseModel):
    member_seq: int
    content: str