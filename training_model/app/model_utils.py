import joblib
import numpy as np

def load_model():
    return joblib.load("model/catboost_model.pkl")

def make_prediction(model, features):
    data = [[
        features.number_of_rooms,
        features.district,
        features.structure_type,
        features.year_of_construction,
        features.floor,
        features.area,
        features.quality
    ]]
    prediction = int(model.predict(data)[0])
    return ["Low", "Medium", "High"][prediction]
