from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle
import numpy as np

# Model aur features load karo
with open('C:/Users/kmpri/Desktop/credit-risk-predictor/src/model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('C:/Users/kmpri/Desktop/credit-risk-predictor/src/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

app = FastAPI()

class LoanApplication(BaseModel):
    EXT_SOURCE_2: float
    EXT_SOURCE_3: float
    AGE_YEARS: float
    YEARS_EMPLOYED: float
    DEBT_INCOME_RATIO: float
    CREDIT_TERM: float

@app.get("/")
def home():
    return {"message": "Credit Risk API is running!"}

@app.post("/predict")
def predict(data: LoanApplication):
    # Saare features ke saath empty dataframe banao
    input_df = pd.DataFrame([np.zeros(len(feature_names))], columns=feature_names)
    
    # Sirf jo features mile unhe fill karo
    input_df['EXT_SOURCE_2'] = data.EXT_SOURCE_2
    input_df['EXT_SOURCE_3'] = data.EXT_SOURCE_3
    input_df['AGE_YEARS'] = data.AGE_YEARS
    input_df['YEARS_EMPLOYED'] = data.YEARS_EMPLOYED
    input_df['DEBT_INCOME_RATIO'] = data.DEBT_INCOME_RATIO
    input_df['CREDIT_TERM'] = data.CREDIT_TERM
    
    prob = model.predict_proba(input_df)[:,1][0]
    decision = "REJECT" if prob >= 0.1 else "APPROVE"
    
    return {
        "default_probability": round(float(prob), 3),
        "decision": decision
    }