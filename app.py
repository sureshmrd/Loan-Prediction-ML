from fastapi import FastAPI # type: ignore
from pydantic import BaseModel # type: ignore
import joblib 
import numpy as np
import uvicorn # type: ignore

class InputData(BaseModel):
    Marriage:int
    Education:int
    Income:int
    LoanAmount:float
    CreditHistory:float


scaler = joblib.load("Scaler.pkl")
model = joblib.load("model.pkl")

app=FastAPI()

@app.post("/predict/")
def predict(input_data : InputData):
    x_values = np.array([[
        input_data.Marriage,
        input_data.Education,
        input_data.Income,
        input_data.LoanAmount,
        input_data.CreditHistory
    ]])

    scaled_values = scaler.transform(x_values)

    prediction = model.predict(scaled_values)

    prediction = int(prediction[0])
    return {"prediction":prediction}

if __name__ == '__main__':
    uvicorn.run(app,host="127.0.0.1",port=8000)


