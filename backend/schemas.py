from pydantic import BaseModel,Field
class Position(BaseModel):
    ticker:str=Field(description="...")
    shares:int=Field(gt=0,description="...")
class LiquidationRequest(BaseModel):
    positions:list[Position]=Field(min_length=1, description="...")
    participation_limit:float=Field(default=0.1,gt=0.0,lt=1.0,description="...")


