from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import plotly.graph_objs as go
import plotly.offline as po

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

CSV_PATH = "csgo_datos_limpios.csv"
df = pd.read_csv(CSV_PATH, low_memory=False)

X_col = "MatchHeadshots"
y_col = "MatchKills"

df[X_col] = pd.to_numeric(df[X_col], errors="coerce")
df[y_col] = pd.to_numeric(df[y_col], errors="coerce")
df = df.dropna(subset=[X_col, y_col])
df = df[(df[X_col] > 0) & (df[y_col] > 0)]

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    data = df.head(10).to_dict(orient="records")
    columns = df.columns.tolist()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "data": data,
        "columns": columns
    })

@app.get("/regresion-lineal-simple", response_class=HTMLResponse)
async def regresion_lineal_simple(request: Request):
    X = df[[X_col]]
    y = df[y_col]

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    # Convertir a numpy arrays planos para evitar problemas con pandas Series
    x_vals = X[X_col].values.flatten()
    y_vals = y.values.flatten()
    y_pred_vals = y_pred.flatten()

    print(f"x_vals sample: {x_vals[:5]}")
    print(f"y_vals sample: {y_vals[:5]}")
    print(f"y_pred_vals sample: {y_pred_vals[:5]}")

    trace_real = go.Scatter(
        x=x_vals,
        y=y_vals,
        mode='markers',
        name='Datos Reales',
        marker=dict(color='orange', size=7, opacity=0.7),
        hovertemplate='Predictor: %{x}<br>Real: %{y}<extra></extra>'
    )
    trace_pred = go.Scatter(
        x=x_vals,
        y=y_pred_vals,
        mode='lines',
        name='Regresión Lineal',
        line=dict(color='red', dash='dash')
    )

    layout = go.Layout(
        title=f'Regresión Lineal Simple: {X_col} → {y_col}',
        xaxis=dict(title=X_col),
        yaxis=dict(title=y_col),
        plot_bgcolor='rgba(20,20,20,0.9)',
        paper_bgcolor='rgba(20,20,20,0.9)',
        font=dict(color='white')
    )
    fig = go.Figure(data=[trace_real, trace_pred], layout=layout)

    # Incluye el script de Plotly con include_plotlyjs=True para evitar problemas
    graph_html = po.plot(fig, include_plotlyjs=True, output_type='div')

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    estadisticas = df.describe().T.reset_index()
    estadisticas_records = estadisticas.to_dict(orient="records")

    return templates.TemplateResponse("regresion_lineal_simple_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })
