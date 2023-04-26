from io import StringIO
from typing import Optional

from fastapi import FastAPI, Request, UploadFile
from .PredictionModel import Model
from fastapi.templating import Jinja2Templates
import pandas as pd
from joblib import load
from .DataModel import DataModel
from .PredictionModel import Model
import os
import nltk
nltk.data.path.append(os.path.abspath('./nltk_data'))
nltk.download('punkt', quiet=True, raise_on_error=True, download_dir='./nltk_data')
nltk.download('stopwords', quiet=True, raise_on_error=True, download_dir='./nltk_data')
nltk.download('wordnet', quiet=True, raise_on_error=True, download_dir='./nltk_data')
ROOT_DIR = os.path.abspath(os.curdir)
NLTK_DATA_DIR = os.path.join(ROOT_DIR, 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)


app = FastAPI()
templates = Jinja2Templates(directory="api/templates")

@app.on_event("startup")
async def startup_event():
    global prediction_model
    prediction_model = Model()

@app.get("/")
def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}


@app.post("/predict")
async def make_predictions(request: Request, file: UploadFile):

    df = pd.read_csv(file.file, sep=',', encoding='utf-8', index_col=0)

    model = Model()
    predictions_list = model.make_predictions(df)
    predictions = predictions_list['prediction']
    return templates.TemplateResponse("index.html", {"request": request, "predictions": predictions})


@app.post("/predictFeeling")
async def predict(request: Request, data):
    model = Model()
    prediction = model.predict(data)
    return prediction

    
