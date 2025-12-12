from fastapi import FastAPI
from app.schemas import PredictRequest, PredictResponse
from app.model import load_model, CLASS_NAMES

app = FastAPI(title="ML FastAPI API", version="1.0.0")

_model = load_model()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    x = [[req.sepal_length, req.sepal_width, req.petal_length, req.petal_width]]

    proba = _model.predict_proba(x)[0]
    class_id = int(proba.argmax())
    predicted_class = CLASS_NAMES[class_id]

    probabilities = {CLASS_NAMES[i]: float(proba[i]) for i in range(len(CLASS_NAMES))}

    return PredictResponse(
        predicted_class=predicted_class,
        class_id=class_id,
        probabilities=probabilities,
    )
