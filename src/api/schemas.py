# purpose is to validate the input from the user befor even usign or calling the model 

from pydantic import BaseModel , Field
from typing import List

class TitanicInput(BaseModel) :
    
    Pclass: int = Field(..., ge=1, le=3, description="Ticket class (1, 2, or 3)")
    Sex: str = Field(..., description="Gender of the passenger")
    Age: float = Field(..., ge=0, description="Age of the passenger")
    SibSp: int = Field(..., ge=0, description="Number of siblings/spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents/children aboard")
    Fare: float = Field(..., ge=0, description="Passenger fare")
    Embarked: str = Field(..., description="Port of Embarkation")

class TitanicOutput(BaseModel):
    prediction: int = Field(..., description="Survival prediction (0 = No, 1 = Yes)")
    probability: float = Field(..., description="Model confidence score")
    status: str = Field("success", description="Request status")    

class TitanicBatchInput(BaseModel):
    inputs: List[TitanicInput]

class TitanicBatchOutput(BaseModel):
    results: List[TitanicOutput]