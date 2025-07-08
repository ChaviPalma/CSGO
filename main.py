from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import json
import os

app = FastAPI(title="CS:GO Analytics")

# Configuración de archivos estáticos y templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

def load_model_stats(model_name: str):
    """Carga las estadísticas desde el JSON generado por el notebook"""
    try:
        with open(f'model_stats/{model_name}_stats.json') as f:
            return json.load(f)
    except FileNotFoundError:
        return {"error": "Modelo no encontrado"}

# Rutas
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("base.html", {"request": request})

@app.get("/svm", response_class=HTMLResponse)
async def svm_page(request: Request):
    stats = load_model_stats("svm")
    return templates.TemplateResponse("svm.html", {
        "request": request,
        "stats": stats,
        "model": "Support Vector Machine"
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)