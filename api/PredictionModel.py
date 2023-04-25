import os
import dill
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import re 
from nltk.tokenize import TweetTokenizer
import os
import nltk
nltk.data.path.append(os.path.abspath('./nltk_data'))
nltk.download('punkt', quiet=True, raise_on_error=True, download_dir='./nltk_data')
nltk.download('stopwords', quiet=True, raise_on_error=True, download_dir='./nltk_data')
nltk.download('wordnet', quiet=True, raise_on_error=True, download_dir='./nltk_data')
ROOT_DIR = os.path.abspath(os.curdir)
NLTK_DATA_DIR = os.path.join(ROOT_DIR, 'nltk_data')
nltk.data.path.append(NLTK_DATA_DIR)



class Model:
    def __init__(self):
        model_path = os.path.join(os.getcwd(), "api/assets", "logistic_regression.pkl")
        
        if not os.path.exists(model_path):
            raise Exception(f"Model file not found in {model_path}")
        with open(model_path, 'rb') as f:
            self.model= dill.load(f)
        if self.model is None:
            raise Exception("Failed to load model")

    def make_predictions(self, data: pd.DataFrame):
        # result = self.model.predict(data)
        
        dic = {}
        for review in data["review_es"]:
            result = self.model.predict([review])

            if (result[0] == 1):
                result = "Positivo"
            else:
                result = "Negativo"

            dic[review] = result
            

        return {"prediction": dic}

    def retrain_model(self, data):
        X = data[["review_es"]]
        Y = data[["sentimiento"]]
        self.model.fit(X, Y)
        y_pred = self.model.predict(X)
        dill.dump(self.model, "assets/logistic_regression.joblib")

        # Compare Y and self.model.predict(X) to see how good the model is
        rmse = mean_squared_error(Y, y_pred, squared=False)
        mae = mean_absolute_error(Y, y_pred)
        r2 = r2_score(Y, y_pred)

        return {"rmse": rmse, "mae": mae, "r2": r2}
    
    