from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
import io
from PIL import Image
import uvicorn
from model import load_model, predict_from_bytes

app = FastAPI(title="Points Checker")

# Load model on startup
model = None

@app.on_event("startup")
async def startup_event():
    global model
    try:
        model = load_model()
    except Exception as e:
        print(f"Error loading model: {str(e)}")

# Create HTML template
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        result = predict_from_bytes(model, contents)
        
        return {
            "prediction": result,
            "message": "The points belong to the same figure!" if result else "The points belong to different figures."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)