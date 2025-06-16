# backend/main.py

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from classifier import classify_query
from database import init_db, log_result

# Initialize FastAPI
app = FastAPI()
init_db()

# CORS config to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Define input model
class Query(BaseModel):
    text: str

# Classification endpoint
@app.post("/classify")
def classify(query: Query):
    label, confidence = classify_query(query.text)
    log_result(query.text, label, confidence)
    return {"intent": label, "confidence": confidence}
