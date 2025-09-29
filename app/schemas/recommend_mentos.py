from pydantic import BaseModel

class RecommendData(BaseModel):
    mentos_seq:int
    mentos_image: str
    mentos_title: str
    price: int
    mento_profile_image: str | None
    region: str