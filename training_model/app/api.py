from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Define input schema
class ApartmentInput(BaseModel):
    number_of_rooms: float
    district: int
    structure_type: int
    year_of_construction: float
    floor: float
    area: float
    quality: float

# Load model
model = joblib.load("../training_model/model/catboost_apartment_price_model.pkl")

# Create FastAPI app
app = FastAPI()

@app.post("/predict")
def predict_price(input_data: ApartmentInput):
    data = np.array([[input_data.number_of_rooms, input_data.district,
                      input_data.structure_type, input_data.year_of_construction,
                      input_data.floor, input_data.area, input_data.quality]])
    prediction = model.predict(data)[0]
    return {"predicted_price": float(prediction)}  # ✅ consistent key
