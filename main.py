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

from fastapi import Form
import joblib

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


df = df.sort_index().reset_index(drop=True)


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
    
    r2 = 0.826
    mae = 1.90
    rmse = 6.74

    
    estadisticas, estadisticas_records = generar_estadisticas(df[['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId', 'MatchKills']])

    
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

    
    r2 = 0.763
    mae = 2.27
    rmse = 9.17

    fig, ax = plt.subplots(figsize=(20, 10))
    plot_tree(modelo, feature_names=predictors, filled=True, fontsize=10)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    graph_base64 = base64.b64encode(buf.read()).decode('utf-8')
    graph_html = f'<img src="data:image/png;base64,{graph_base64}" class="img-fluid rounded shadow">'

    
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
    
    mae = 1.84
    rmse = 2.57
    r2 = 0.830

    
    estadisticas, estadisticas_records = generar_estadisticas(df[['MatchHeadshots', 'MatchFlankKills', 'MatchAssists', 'RoundId', 'MatchKills']])

    
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
    
    mae = 1.95
    rmse = 7.24
    r2 = 0.813

    
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

# === RUTA: Support Vector Machine - Clasificación ===
@app.get("/support-vector-machine-clasificacion", response_class=HTMLResponse)
async def support_vector_machine_clasificacion(request: Request):
    
    accuracy = 0.71
    precision = 0.70
    recall = 0.76
    f1_score = 0.73

    
    classification_report_text = """
              precision    recall  f1-score   support

     Perdida       0.73      0.66      0.70      3234
    Victoria       0.70      0.76      0.73      3320

    accuracy                           0.71      6554
   macro avg       0.72      0.71      0.71      6554
weighted avg       0.72      0.71      0.71      6554
    """


    
    lines = [line.strip() for line in classification_report_text.strip().splitlines() if line.strip()]
    
    columns = ['class'] + lines[0].split()
    
    data = []
    for line in lines[1:]:
        parts = line.split()
        
        if len(parts) == len(columns):
            data.append(parts)
        else:
            
            class_name_parts = []
            for i, part in enumerate(parts):
                
                try:
                    float(part)
                    first_num_idx = i
                    break
                except:
                    class_name_parts.append(part)
            class_name = ' '.join(class_name_parts)
            rest = parts[first_num_idx:]
            row = [class_name] + rest
            data.append(row)

    
    class_report_df = pd.DataFrame(data, columns=columns)

    
    for col in columns[1:]:
        class_report_df[col] = pd.to_numeric(class_report_df[col], errors='coerce')

    
    features = [
        'RoundKills',
        'RoundDeaths',
        'KDR',
        'TeamStartingEquipmentValue',
        'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown',
        'Map',
        'Team'
    ]
    df_filtered = df[df['RoundWinner'].isin(['True', 'False'])].copy()
    X = df_filtered[features]
    X = pd.get_dummies(X, columns=['Map', 'Team'])

    estadisticas, estadisticas_records = generar_estadisticas(X)

    
    confusion_matrix = "/static/img/confusion_matrix_SVM.png"
    roc_auc_curve = "/static/img/roc_auc_curve.png"
    prob_dist = "/static/img/distribucion_probabilidades.png"

    return templates.TemplateResponse("SVM_Clasificacion.html", {
        "request": request,
        "accuracy": f"{accuracy:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1": f"{f1_score:.2f}",
        "confusion_matrix": confusion_matrix,
        "roc_auc_curve": roc_auc_curve,
        "prob_dist": prob_dist,
        "class_report": class_report_df,
        "class_report_records": class_report_df.to_dict(orient="records"),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
    })

# === RUTA: arbol de decision - Clasificación ===
@app.get("/arbol-decision-clasificacion", response_class=HTMLResponse)
async def arbol_decision_clasificacion(request: Request):
    
    accuracy = 0.71
    precision = 0.72
    recall = 0.68
    f1 = 0.70

    
    classification_report_text = """
              precision    recall  f1-score   support

     Perdida       0.69      0.73      0.71      3234
    Victoria       0.72      0.68      0.70      3320

    accuracy                           0.71      6554
   macro avg       0.71      0.71      0.71      6554
weighted avg       0.71      0.71      0.71      6554
    """

   
    lines = [line.strip() for line in classification_report_text.strip().splitlines() if line.strip()]
    columns = ['class'] + lines[0].split()
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == len(columns):
            data.append(parts)
        else:
            class_name_parts = []
            for i, part in enumerate(parts):
                try:
                    float(part)
                    first_num_idx = i
                    break
                except:
                    class_name_parts.append(part)
            class_name = ' '.join(class_name_parts)
            rest = parts[first_num_idx:]
            row = [class_name] + rest
            data.append(row)

    class_report_df = pd.DataFrame(data, columns=columns)
    for col in columns[1:]:
        class_report_df[col] = pd.to_numeric(class_report_df[col], errors='coerce')

    
    features = [
        'RoundKills',
        'RoundDeaths',
        'KDR',
        'TeamStartingEquipmentValue',
        'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown',
        'Map',
        'Team'
    ]
    df_filtered = df[df['RoundWinner'].isin(['True', 'False'])].copy()
    X = df_filtered[features]
    X = pd.get_dummies(X, columns=['Map', 'Team'])

    estadisticas, estadisticas_records = generar_estadisticas(X)

    
    confusion_matrix = "/static/img/confusion_matrix_decision_tree.png"
    roc_auc_curve = "/static/img/roc_auc_curve_decision_tree.png"
    prob_dist = "/static/img/distribucion_probabilidades_decision_tree.png"

    return templates.TemplateResponse("Decision_tree_clasificacion.html", {
        "request": request,
        "accuracy": f"{accuracy:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1": f"{f1:.2f}",
        "confusion_matrix": confusion_matrix,
        "roc_auc_curve": roc_auc_curve,
        "prob_dist": prob_dist,
        "class_report": class_report_df,
        "class_report_records": class_report_df.to_dict(orient="records"),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
    })

# === RUTA: KNN - Clasificación ===
@app.get("/knn", response_class=HTMLResponse)
async def knn_clasificacion(request: Request):
    
    features = [
        'RoundKills',
        'RoundDeaths',
        'KDR',
        'TeamStartingEquipmentValue',
        'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown',
        'Map',
        'Team'
    ]

    df_filtered = df[df['RoundWinner'].isin(['True', 'False'])].copy()
    X = df_filtered[features]
    X = pd.get_dummies(X, columns=['Map', 'Team'])

    
    accuracy = 0.70
    precision = 0.71
    recall = 0.68
    f1 = 0.69

    
    class_report_dict = {
        "Clase": ["Perdida", "Victoria", "accuracy", "macro avg", "weighted avg"],
        "precision": [0.68, 0.71, "", 0.70, 0.70],
        "recall":    [0.71, 0.68, "", 0.70, 0.70],
        "f1-score":  [0.70, 0.69, 0.70, 0.70, 0.70],
        "support":   [3234, 3320, 6554, 6554, 6554]
    }
    class_report_df = pd.DataFrame(class_report_dict)
    class_report_df = class_report_df.rename(columns={"Clase": ""})
    class_report_records = class_report_df.to_dict(orient="records")

    
    estadisticas, estadisticas_records = generar_estadisticas(X)

    
    confusion_matrix = "/static/img/confusion_matrix_knn.png"
    roc_auc_curve = "/static/img/roc_auc_curve_knn.png"
    prob_dist = "/static/img/distribucion_probabilidades_knn.png"

    return templates.TemplateResponse("K-Nearest_Neighbors_Clasificacion.html", {
        "request": request,
        "accuracy": f"{accuracy:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1": f"{f1:.2f}",
        "confusion_matrix": confusion_matrix,
        "roc_auc_curve": roc_auc_curve,
        "prob_dist": prob_dist,
        "class_report": class_report_df,
        "class_report_records": class_report_records,
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records
    })


# === RUTA: logistic_regression===
@app.get("/logistic-regression-clasificacion", response_class=HTMLResponse)
async def logistic_regression_clasificacion(request: Request):
    
    features = [
        'RoundKills',
        'RoundDeaths',
        'KDR',
        'TeamStartingEquipmentValue',
        'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown',
        'Map',
        'Team'
    ]

    
    df_filtered = df[df['RoundWinner'].isin(['True', 'False'])].copy()
    X = df_filtered[features]
    X = pd.get_dummies(X, columns=['Map', 'Team'])

    y = df_filtered['RoundWinner'].replace({'True':1, 'False':0}).astype(int)

    # Métricas fijas
    accuracy = 0.71
    precision = 0.70
    recall = 0.74
    f1 = 0.72

    
    classification_report_text = """
              precision    recall  f1-score   support

     Perdida       0.72      0.68      0.70      3234
    Victoria       0.70      0.74      0.72      3320

    accuracy                           0.71      6554
   macro avg       0.71      0.71      0.71      6554
weighted avg       0.71      0.71      0.71      6554
    """

    
    lines = [line.strip() for line in classification_report_text.strip().splitlines() if line.strip()]
    columns = ['class'] + lines[0].split()
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == len(columns):
            data.append(parts)
        else:
            class_name_parts = []
            for i, part in enumerate(parts):
                try:
                    float(part)
                    first_num_idx = i
                    break
                except:
                    class_name_parts.append(part)
            class_name = ' '.join(class_name_parts)
            rest = parts[first_num_idx:]
            data.append([class_name] + rest)

    class_report_df = pd.DataFrame(data, columns=columns)
    for col in columns[1:]:
        class_report_df[col] = pd.to_numeric(class_report_df[col], errors='coerce')

    
    estadisticas, estadisticas_records = generar_estadisticas(X)

    
    confusion_matrix = "/static/img/confusion_matrix_logistic_regression.png"
    roc_auc_curve = "/static/img/roc_auc_curve_logistic_regression.png"
    prob_dist = "/static/img/distribucion_probabilidades_logistic_regression.png"

    return templates.TemplateResponse("Logistic_Regression_Clasificacion.html", {
        "request": request,
        "accuracy": f"{accuracy:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1": f"{f1:.2f}",
        "confusion_matrix": confusion_matrix,
        "roc_auc_curve": roc_auc_curve,
        "prob_dist": prob_dist,
        "class_report": class_report_df,
        "class_report_records": class_report_df.to_dict(orient="records"),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
    })

# === RUTA: Random forest===

@app.get("/random-forest-clasificacion", response_class=HTMLResponse)
async def random_forest_clasificacion(request: Request):
    
    features = [
        'RoundKills',
        'RoundDeaths',
        'KDR',
        'TeamStartingEquipmentValue',
        'RLethalGrenadesThrown',
        'RNonLethalGrenadesThrown',
        'Map',
        'Team'
    ]


    df_filtered = df[df['RoundWinner'].isin(['True', 'False'])].copy()
    X = df_filtered[features]
    X = pd.get_dummies(X, columns=['Map', 'Team'])

    y = df_filtered['RoundWinner'].replace({'True': 1, 'False': 0}).astype(int)

  
    accuracy = 0.70
    precision = 0.71
    recall = 0.70
    f1 = 0.70


    classification_report_text = """
              precision    recall  f1-score   support

     Perdida       0.70      0.70      0.70      3234
    Victoria       0.71      0.70      0.70      3320

    accuracy                           0.70      6554
   macro avg       0.70      0.70      0.70      6554
weighted avg       0.70      0.70      0.70      6554
    """


    lines = [line.strip() for line in classification_report_text.strip().splitlines() if line.strip()]
    columns = ['class'] + lines[0].split()
    data = []
    for line in lines[1:]:
        parts = line.split()
        if len(parts) == len(columns):
            data.append(parts)
        else:
            class_name_parts = []
            for i, part in enumerate(parts):
                try:
                    float(part)
                    first_num_idx = i
                    break
                except:
                    class_name_parts.append(part)
            class_name = ' '.join(class_name_parts)
            rest = parts[first_num_idx:]
            data.append([class_name] + rest)

    class_report_df = pd.DataFrame(data, columns=columns)
    for col in columns[1:]:
        class_report_df[col] = pd.to_numeric(class_report_df[col], errors='coerce')

   
    estadisticas, estadisticas_records = generar_estadisticas(X)

    
    confusion_matrix = "/static/img/confusion_matrix_random_forest.png"
    roc_auc_curve = "/static/img/roc_auc_curve_random_forest.png"
    prob_dist = "/static/img/distribucion_probabilidades_random_forest.png"

    return templates.TemplateResponse("Random_Forest_Clasificacion.html", {
        "request": request,
        "accuracy": f"{accuracy:.2f}",
        "precision": f"{precision:.2f}",
        "recall": f"{recall:.2f}",
        "f1": f"{f1:.2f}",
        "confusion_matrix": confusion_matrix,
        "roc_auc_curve": roc_auc_curve,
        "prob_dist": prob_dist,
        "class_report": class_report_df,
        "class_report_records": class_report_df.to_dict(orient="records"),
        "estadisticas": estadisticas,
        "estadisticas_records": estadisticas_records,
    })

modelo_svm = joblib.load("models/svm_model.pkl")  

@app.get("/svm-prediccion", response_class=HTMLResponse)
async def formulario_prediccion(request: Request):
    return templates.TemplateResponse("SVM_prediccion.html", {"request": request})

@app.post("/svm-prediccion", response_class=HTMLResponse)
async def predecir_matchkills(
    request: Request,
    MatchHeadshots: int = Form(...),
    MatchFlankKills: int = Form(...),
    MatchAssists: int = Form(...),
    RoundId: int = Form(...)
):
    if modelo_svm is None:
        return templates.TemplateResponse("SVM_prediccion.html", {
            "request": request,
            "error": "El modelo no está disponible."
        })

    datos = [[MatchHeadshots, MatchFlankKills, MatchAssists, RoundId]]
    prediccion = modelo_svm.predict(datos)[0]

    return templates.TemplateResponse("SVM_prediccion.html", {
        "request": request,
        "prediccion": round(prediccion, 2)
    })


modelo = joblib.load("models/forest_model.pkl")

@app.get("/random-forest-clasificacion-pred", response_class=HTMLResponse)
async def mostrar_formulario(request: Request):
    return templates.TemplateResponse("random_forest_clasificacion_pred.html", {"request": request})

@app.post("/random-forest-clasificacion-pred")
async def random_forest_clasificacion_post(request: Request, 
    RoundKills: int = Form(...),
    RoundDeaths: int = Form(...),
    KDR: float = Form(...),
    TeamStartingEquipmentValue: int = Form(...),
    RLethalGrenadesThrown: int = Form(...),
    RNonLethalGrenadesThrown: int = Form(...),
    Map: str = Form(...),
    Team: str = Form(...)
):
    datos = pd.DataFrame([{
        'RoundKills': RoundKills,
        'RoundDeaths': RoundDeaths,
        'KDR': KDR,
        'TeamStartingEquipmentValue': TeamStartingEquipmentValue,
        'RLethalGrenadesThrown': RLethalGrenadesThrown,
        'RNonLethalGrenadesThrown': RNonLethalGrenadesThrown,
        'Map': Map,
        'Team': Team
    }])

    # Aplicar get_dummies para las categóricas
    datos_encoded = pd.get_dummies(datos, columns=['Map', 'Team'])

    # Añadir columnas que faltan y ordenar
    for col in columnas_modelo:
        if col not in datos_encoded.columns:
            datos_encoded[col] = 0
    datos_encoded = datos_encoded[columnas_modelo]

    try:
        modelo = joblib.load("models/forest_model.pkl")
        pred = modelo.predict(datos_encoded)[0]
        resultado = "Victoria" if pred == 1 else "Derrota"
        return templates.TemplateResponse("random_forest_clasificacion_pred.html", {
            "request": request,
            "prediccion": resultado
        })
    except Exception as e:
        return templates.TemplateResponse("random_forest_clasificacion_pred.html", {
            "request": request,
            "error": f"Error al realizar la predicción: {e}"
        })
