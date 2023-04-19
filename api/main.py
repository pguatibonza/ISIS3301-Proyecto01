from io import StringIO
from typing import Optional

from fastapi import FastAPI, Request, UploadFile
from PredictionModel import Model
from fastapi.templating import Jinja2Templates
import pandas as pd
from joblib import load
from DataModel import DataModel

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
async def make_predictions(request: Request, file: UploadFile):

    df = pd.read_csv(file.file, sep=',', encoding='utf-8', index_col=0)

    model = Model(columns=df.columns.tolist())
    predictions_dic = model.make_predictions(df)
    predictions = predictions_dic['prediction']
    return templates.TemplateResponse("index.html", {"request": request, "predictions": dict(predictions)})


@app.post("/retrain")
async def retrain_model(request: Request, file: UploadFile):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode()))

    model = Model()
    retrain = model.retrain_model(df)
    return templates.TemplateResponse("index.html", {"request": request, "rmse": retrain[0], "mae": retrain[1], "r2": retrain[2]})
