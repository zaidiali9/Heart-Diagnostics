from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
app = FastAPI()

class InputData(BaseModel):
    Smoking: int
    Stroke: int
    PhysicalHealth: float
    DiffWalking: int
    Sex: int
    AgeCategory: int
    Diabetic: int
    PhysicalActivity: int
    KidneyDisease: int
    SkinCancer: int


with open('modelfile.pkl', 'rb') as file:
    model = pickle.load(file)

@app.get("/")
async def read_root(items:InputData):
        df = pd.DataFrame([InputData.dict().values()], columns=InputData.dict().keys())
        yhat = model.predict(df)
        return {"prediction": int(yhat)}