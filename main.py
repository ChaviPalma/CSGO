from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

import plotly.graph_objs as go
import plotly.offline as po

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import io
import base64

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === Cargar y preparar datos ===
CSV_PATH = "csgo_datos_limpios.csv"
df = pd.read_csv(CSV_PATH, low_memory=False)

# Convertir columnas necesarias a numéricas
variables = ['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId', 'MatchKills']
for var in variables:
    df[var] = pd.to_numeric(df[var], errors="coerce")
df = df.dropna(subset=variables)
df = df[(df["MatchHeadshots"] > 0) & (df["MatchKills"] > 0)]

# ✅ CORRECCIÓN: garantizar mismo orden de datos que en el notebook
df = df.sort_index().reset_index(drop=True)

# Función para estadísticas descriptivas
def generar_estadisticas(df_subset):
    estadisticas = df_subset.describe().T.reset_index()
    estadisticas_records = estadisticas.to_dict(orient="records")
    return estadisticas, estadisticas_records

# === RUTA: Página de inicio ===
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    data = df.head(10).to_dict(orient="records")
    columns = df.columns.tolist()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "data": data,
        "columns": columns
    })

# === RUTA: Regresión Lineal Simple ===
@app.get("/regresion-lineal-simple", response_class=HTMLResponse)
async def regresion_lineal_simple(request: Request):
    X_col = "MatchHeadshots"
    y_col = "MatchKills"
    X = df[[X_col]]
    y = df[y_col]

   
    r2 = 0.692
    mae = 2.63
    rmse = 3.46

    # Definir estadisticas y estadisticas_records
    estadisticas, estadisticas_records = generar_estadisticas(df[[X_col, y_col]])

    return templates.TemplateResponse("regresion_lineal_simple_modelo_regresion.html", {
        "request": request,
        "r2_score": f"{r2:.3f}",  # string formateado
        "mae": f"{mae:.2f}",
        "rmse": f"{rmse:.2f}",
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_path": "/static/img/regresion_lineal_simple.png"
    })

# === RUTA: Regresión Lineal Múltiple ===
@app.get("/regresion-lineal-multiple", response_class=HTMLResponse)
async def regresion_lineal_multiple(request: Request):
    # Métricas fijas que quieres mostrar
    r2 = 0.826
    mae = 1.90
    rmse = 6.74

    # Estadísticas descriptivas del dataset completo (puedes ajustar columnas si quieres)
    estadisticas, estadisticas_records = generar_estadisticas(df[['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId', 'MatchKills']])

    # Ruta imagen estática que ya debes tener en /static/img/
    graph_path = "/static/img/regresion_lineal_multiple.png"

    return templates.TemplateResponse("Regresion_lineal_multiple_modelo_regresion.html", {
        "request": request,
        "r2_score": f"{r2:.3f}",
        "mae": f"{mae:.2f}",
        "rmse": f"{rmse:.2f}",
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_path": graph_path
    })



# === RUTA: Decision tree ===
@app.get("/arbol-decision-regresion", response_class=HTMLResponse)
async def arbol_decision_regresion(request: Request):
    # Variables seleccionadas (como en el notebook)
    predictors = ['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'MatchWinner']
    target = 'MatchKills'

    # Convertir MatchWinner a 0/1
    df['MatchWinner'] = df['MatchWinner'].astype(str).map({'True': 1, 'False': 0})

    X = df[predictors]
    y = df[target]

    # Entrenamiento
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    modelo = DecisionTreeRegressor(max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas fijas que quieres mostrar
    r2 = 0.763
    mae = 2.27
    rmse = 9.17

    # Graficar árbol y codificar en base64
    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(modelo, feature_names=predictors, filled=True, fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    graph_html = f'<img src="data:image/png;base64,{graph_base64}" class="img-fluid rounded shadow">'

    # Estadísticas descriptivas
    estadisticas, estadisticas_records = generar_estadisticas(pd.concat([X_test, y_test], axis=1))

    return templates.TemplateResponse("Decision_tree_modelo_regresion.html", {
        "request": request,
        "r2_score": f"{r2:.3f}",
        "mae": f"{mae:.2f}",
        "rmse": f"{rmse:.2f}",
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })


# === RUTA: Support Vector Machine ===
@app.get("/support-vector-machine-regresion", response_class=HTMLResponse)
async def support_vector_machine_regresion(request: Request):
    # Métricas fijas que quieres mostrar
    mae = 1.84
    rmse = 2.57
    r2 = 0.830

    # Estadísticas descriptivas (puedes ajustar según tu dataset)
    estadisticas, estadisticas_records = generar_estadisticas(df[['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId', 'MatchKills']])

    # Ruta de la imagen estática que entregas
    graph_path = "/static/img/svm_regresion.png"

    return templates.TemplateResponse("Support_Vector_Machine_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 3),
        "mae": round(mae, 3),
        "rmse": round(rmse, 3),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_path": graph_path
    })

# === RUTA: Random Forest Regresión ===
@app.get("/random-forest-regresion", response_class=HTMLResponse)
async def random_forest_regresion(request: Request):
    # Métricas fijas que quieres mostrar
    mae = 1.95
    rmse = 7.24
    r2 = 0.813

    # Estadísticas descriptivas (ajustar según dataset)
    estadisticas, estadisticas_records = generar_estadisticas(df[['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId', 'MatchKills']])

    graph_path = "/static/img/random_forest_regresion.png"

    return templates.TemplateResponse("Random _forest_modelo_regresion.html", {
        "request": request,
        "r2_score": f"{r2:.3f}",
        "mae": f"{mae:.2f}",
        "rmse": f"{rmse:.2f}",
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_path": graph_path
    })

# === Modelos de Clasificación===