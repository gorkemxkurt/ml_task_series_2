# ML FastAPI API

## English

A simple FastAPI service that exposes a machine learning prediction endpoint.

### What this project does
- Runs a local FastAPI server
- Provides a health-check endpoint (`/health`)
- Provides a prediction endpoint (`/predict`) that returns a predicted class and probabilities

### Endpoints
- **GET** `/health` → returns `{"status": "ok"}`
- **POST** `/predict` → returns `predicted_class`, `class_id`, `probabilities`

### How to run (PyCharm friendly)
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Start the API:
   - `python run_api.py`
3. Open Swagger UI in the browser:
   - `http://127.0.0.1:8000/docs`

### Example request
**POST** `/predict`
```json
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
