import os
import uvicorn
import requests
from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from utils import process_data, process_bug_data

TRAINR_ENDPOINT = os.getenv("TRAINR_ENDPOINT")

# defining the main app
app = FastAPI(title="processr", docs_url="/")

# class which is expected in the payload while training
class DataIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    flower_class: str

class BugDataIn(BaseModel):
    lines_of_code: float
    cyclomatic_complexity: float
    essential_complexity: float
    design_complexity : float
    totalo_perators_operands : float
    volume : float
    program_length : float
    difficulty : float
    intelligence  : float
    effort  : float
    b  : float 
    time_estimator  : float
    lOCode    : float
    lOComment : float
    lOBlank  : float
    lOCodeAndComment: float
    uniq_Op  : float
    uniq_Opnd  : float
    total_Op  : float
    total_Opnd : float
    branchCount : float
    defects : bool
    

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}


@app.post("/processBug", status_code=200)
# Route to take in data, process it and send it for training.
def processBug(data: List[BugDataIn]):
    processed = process_bug_data(data)
    # send the processed data to trainr for training
    response = requests.post(f"{TRAINR_ENDPOINT}/trainBug", json=processed)
    return {"detail": "Processing successful"}


@app.post("/process", status_code=200)
# Route to take in data, process it and send it for training.
def process(data: List[DataIn]):
    processed = process_data(data)
    # send the processed data to trainr for training
    response = requests.post(f"{TRAINR_ENDPOINT}/train", json=processed)
    return {"detail": "Processing successful"}
    
    
# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=8888, reload=True)
