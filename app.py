from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib

# Initialize the FastAPI application
app = FastAPI(title="Bank Customer Churn Prediction")


# Load the pipeline
pipeline = joblib.load("best_pipeline_tuned.joblib")


templates = Jinja2Templates(directory="templates")

# Home Route
@app.get("/")
def home(request : Request):
    return templates.TemplateResponse(
        "index.html",
        {"request" : request}   #FastAPI + Jinja requires the request object for rendering templates.
    )
    
# Predict Route - Recieves values submitted from HTML form
@app.post("/predict")
def predict(
        request: Request,
        credit_score : float = Form(...),
        country : str = Form(...),
        gender : str = Form(...),
        age : float = Form(...),
        tenure: float = Form(...),
        balance: float = Form(...),
        products_number: float = Form(...),
        credit_card: float = Form(...),
        active_member: float = Form(...),
        estimated_salary: float = Form(...)
    ):

    # Create a DataFrame with proper column names
    input_data = pd.DataFrame([{
        "credit_score": credit_score,
        "country": country,
        "gender": gender,
        "age": age,
        "tenure": tenure,
        "balance": balance,
        "products_number": products_number,
        "credit_card": credit_card,
        "active_member": active_member,
        "estimated_salary": estimated_salary
    }])

    # Predict churn probability
    pred_prob = pipeline.predict_proba(input_data)[0][1]

    # Threshold to reduce false negatives (catch churners)
    threshold = 0.3
    pred = 1 if pred_prob > threshold else 0

    return templates.TemplateResponse("index.html", {
        "request": request,
        "Prediction": "Churn" if pred == 1 else "No Churn",
        "Probability": round(pred_prob * 100, 2),
        "Threshold": threshold
    })



