from typing import Optional

from fastapi import FastAPI, Request

import pandas as  pd
from joblib import load
from DataModel import DataModel

app = FastAPI()


@app.get("/")
def read_root():
   return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
   return {"item_id": item_id, "q": q}

@app.post("/predict")
def make_predictions(request : Request, dataModel: DataModel):
    df = pd.DataFrame(dataModel.dict(), columns=dataModel.dict().keys(), index=[0])
    df.columns = dataModel.columns()
    model = load("assets/logistic_regression.joblib")
    result = model.predict(df)
    return result

@app.post("/retrain")
def retrain_model(dataModel: DataModel):

   return None