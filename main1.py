from fastapi import FastAPI
import joblib
import pandas as pd
import os

app = FastAPI()  # initialization of FastAPI class

# loading the model
MODEL_PATH = "mymodel.pkl" if os.path.exists("mymodel.pkl") else "model.pkl"
model = joblib.load(MODEL_PATH)

@app.get("/")
def testing():
    return {"test": "all ok"}

@app.post("/prediction")
def myprediction(hours: float):
    newdata = pd.DataFrame({
        "StudyHours": [hours]
    })

    mynewdata = model.predict(newdata)
    pred_val = int(mynewdata[0])

    return {
        "prediction": pred_val,
        "result": "Pass" if pred_val == 1 else "Fail"
    }
