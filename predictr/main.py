import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from utils import load_model, load_bug_model, predict, predict_bug

# defining the main app
app = FastAPI(title="predictr", docs_url="/")

# class which is expected in the payload
class QueryIn(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

class BugQueryIn(BaseModel):
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


# class which is returned in the response
class QueryOut(BaseModel):
    flower_class: str


class BugQueryOut(BaseModel):
    defects : bool

# Route definitions
@app.get("/ping")
# Healthcheck route to ensure that the API is up and running
def ping():
    return {"ping": "pong"}

@app.post("/predict_bug", response_model=BugQueryOut, status_code=200)
def predict_bug(query_data: BugQueryIn):
    output = {"defects": predict_bug(query_data)}
    return output
    
@app.post("/reload_bug_model", status_code=200)
# Route to reload the model from file
def reload_bug_model():
    load_bug_model()
    output = {"detail": "Model successfully loaded"}
    return output
    
    
@app.post("/predict_flower", response_model=QueryOut, status_code=200)
# Route to do the prediction using the ML model defined.
# Payload: QueryIn containing the parameters
# Response: QueryOut containing the flower_class predicted (200)
def predict_flower(query_data: QueryIn):
    output = {"flower_class": predict(query_data)}
    return output


@app.post("/reload_model", status_code=200)
# Route to reload the model from file
def reload_model():
    load_model()
    output = {"detail": "Model successfully loaded"}
    return output


# Main function to start the app when main.py is called
if __name__ == "__main__":
    # Uvicorn is used to run the server and listen for incoming API requests on 0.0.0.0:8888
    uvicorn.run("main:app", host="0.0.0.0", port=9999, reload=True)
