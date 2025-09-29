from pydantic import BaseModel

class Message(BaseModel):
    member_seq: str
    content: str