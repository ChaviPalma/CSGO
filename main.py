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

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    # Gráfico 2D
    trace_real = go.Scatter(
        x=X[X_col],
        y=y,
        mode='markers',
        name='Datos Reales',
        marker=dict(color='orange', size=7, opacity=0.7),
        hovertemplate='Headshots: %{x}<br>Kills: %{y}<extra></extra>'
    )
    trace_pred = go.Scatter(
        x=X[X_col],
        y=y_pred,
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
    graph_html = po.plot(fig, include_plotlyjs=True, output_type='div')

    # Métricas
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    estadisticas, estadisticas_records = generar_estadisticas(df[[X_col, y_col]])

    return templates.TemplateResponse("regresion_lineal_simple_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })

# === RUTA: Regresión Lineal Múltiple ===
@app.get("/regresion-lineal-multiple", response_class=HTMLResponse)
async def regresion_lineal_multiple(request: Request):
    predictors = ['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId']
    target = 'MatchKills'

    X = df[predictors]
    y = df[target]

    # División 80/20 igual que notebook
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = LinearRegression()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Para gráfico 3D usar variables iguales que notebook (headshots y flankkills)
    X_3d = X_test[['MatchHeadshots', 'MatchFlankKills']]
    y_3d = y_test

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_3d["MatchHeadshots"],
        y=X_3d["MatchFlankKills"],
        z=y_3d,
        mode='markers',
        marker=dict(
            size=6,
            color=y_3d,
            colorscale='Viridis',
            opacity=0.9,
            colorbar=dict(title='Kills')
        ),
        name="Datos Test"
    ))

    fig.update_layout(
        title="Regresión Lineal Múltiple (Test): Headshots vs FlankKills vs Kills",
        scene=dict(
            xaxis=dict(title="MatchHeadshots", backgroundcolor="rgb(20,20,20)", gridcolor="gray", showbackground=True, zerolinecolor="white", showspikes=True),
            yaxis=dict(title="MatchFlankKills", backgroundcolor="rgb(20,20,20)", gridcolor="gray", showbackground=True, zerolinecolor="white", showspikes=True),
            zaxis=dict(title="MatchKills", backgroundcolor="rgb(20,20,20)", gridcolor="gray", showbackground=True, zerolinecolor="white", showspikes=True),
        ),
        width=900,
        height=700,
        template='plotly_dark',
        font=dict(color='white')
    )

    graph_html = po.plot(fig, include_plotlyjs=True, output_type="div")
    estadisticas, estadisticas_records = generar_estadisticas(pd.concat([X_test, y_test], axis=1))

    return templates.TemplateResponse("Regresion_lineal_multiple_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })
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

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    # Gráfico 2D
    trace_real = go.Scatter(
        x=X[X_col],
        y=y,
        mode='markers',
        name='Datos Reales',
        marker=dict(color='orange', size=7, opacity=0.7),
        hovertemplate='Headshots: %{x}<br>Kills: %{y}<extra></extra>'
    )
    trace_pred = go.Scatter(
        x=X[X_col],
        y=y_pred,
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
    graph_html = po.plot(fig, include_plotlyjs=True, output_type='div')

    # Métricas
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    estadisticas, estadisticas_records = generar_estadisticas(df[[X_col, y_col]])

    return templates.TemplateResponse("regresion_lineal_simple_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })

# === RUTA: Regresión Lineal Múltiple ===
@app.get("/regresion-lineal-multiple", response_class=HTMLResponse)
async def regresion_lineal_multiple(request: Request):
    predictors = ['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId']
    target = 'MatchKills'

    X = df[predictors]
    y = df[target]

    modelo = LinearRegression()
    modelo.fit(X, y)
    y_pred = modelo.predict(X)

    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    # Gráfico 3D con las dos variables principales
    X_3d = X[['MatchHeadshots', 'MatchFlankKills']]
    y_3d = y

    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=X_3d['MatchHeadshots'],
        y=X_3d['MatchFlankKills'],
        z=y_3d,
        mode='markers',
        marker=dict(
            size=6,
            color=y_3d,
            colorscale='Viridis',
            opacity=0.9,
            colorbar=dict(title='Kills')
        ),
        name="Datos"
    ))

    fig.update_layout(
        title="Regresión Lineal Múltiple: Headshots vs FlankKills vs Kills",
        scene=dict(
            xaxis=dict(title="MatchHeadshots", backgroundcolor="rgb(20,20,20)", gridcolor="gray", showbackground=True, zerolinecolor="white", showspikes=True),
            yaxis=dict(title="MatchFlankKills", backgroundcolor="rgb(20,20,20)", gridcolor="gray", showbackground=True, zerolinecolor="white", showspikes=True),
            zaxis=dict(title="MatchKills", backgroundcolor="rgb(20,20,20)", gridcolor="gray", showbackground=True, zerolinecolor="white", showspikes=True)
        ),
        width=900,
        height=700,
        template='plotly_dark',
        font=dict(color='white')
    )

    graph_html = po.plot(fig, include_plotlyjs=True, output_type="div")
    
    # Generar estadísticas solo con las columnas utilizadas en el modelo para evitar confusión
    estadisticas, estadisticas_records = generar_estadisticas(pd.concat([X, y], axis=1))

    return templates.TemplateResponse("Regresion_lineal_multiple_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
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

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

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
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })

# === RUTA: Support Vector Machine ===
@app.get("/support-vector-machine-regresion", response_class=HTMLResponse)
async def support_vector_machine_regresion(request: Request):
    predictors = ['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId']
    target = 'MatchKills'

    X = df[predictors]
    y = df[target]

    # División entrenamiento/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Entrenamiento modelo SVM
    modelo = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    # Métricas
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Gráfico real vs predicho
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_test,
        y=y_pred,
        mode='markers',
        name='Predicciones',
        marker=dict(color='cyan', size=6, opacity=0.7),
        hovertemplate='Real: %{x}<br>Predicho: %{y}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[y_test.min(), y_test.max()],
        y=[y_test.min(), y_test.max()],
        mode='lines',
        name='Ideal',
        line=dict(color='red', dash='dash')
    ))
    fig.update_layout(
        title="SVM: Valores Reales vs Predichos",
        xaxis_title="Kills Reales",
        yaxis_title="Kills Predichos",
        template="plotly_dark",
        font=dict(color='white'),
        width=800,
        height=600
    )

    graph_html = po.plot(fig, include_plotlyjs=True, output_type="div")
    estadisticas, estadisticas_records = generar_estadisticas(pd.concat([X_test, y_test], axis=1))

    return templates.TemplateResponse("Support_Vector_Machine_modelo_regresion.html", {
        "request": request,
        "r2_score": round(r2, 4),
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
        "graph_html": graph_html
    })
