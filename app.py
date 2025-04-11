from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

app = Flask(__name__)

CSV_FILE = "datos_estudiantes.csv"
IMG_DIR = "static"
MODEL_FILE = "modelo.pkl"

# ----------------------------------------
# UTILIDADES DE DATOS Y MODELO
# ----------------------------------------

def generar_datos(n=200):
    np.random.seed(0)
    data = []

    mitad = n // 2
    # Generar mitad Aprobados
    for i in range(mitad):
        notas = {
            'matematica': round(np.random.uniform(11, 20), 2),
            'comunicacion': round(np.random.uniform(11, 20), 2),
            'ciencia y tecnologia': round(np.random.uniform(11, 20), 2),
            'personal social': round(np.random.uniform(11, 20), 2)
        }
        promedio = round(np.mean(list(notas.values())), 2)
        estudiante = {
            'nombre': f'Estudiante_{i}',
            'asistencia': 100,
            **notas,
            'promedio': promedio,
            'estado': 'Aprobado'
        }
        data.append(estudiante)

    # Generar mitad Desaprobados (algunos ausentes también)
    for i in range(mitad, n):
        presente = np.random.rand() < 0.7
        if not presente:
            estudiante = {
                'nombre': f'Estudiante_{i}',
                'asistencia': 0,
                'matematica': 0.0,
                'comunicacion': 0.0,
                'ciencia y tecnologia': 0.0,
                'personal social': 0.0,
                'promedio': 0.0,
                'estado': 'Desaprobado'
            }
        else:
            notas = {
                'matematica': round(np.random.uniform(0, 10.5), 2),
                'comunicacion': round(np.random.uniform(0, 10.5), 2),
                'ciencia y tecnologia': round(np.random.uniform(0, 10.5), 2),
                'personal social': round(np.random.uniform(0, 10.5), 2)
            }
            promedio = round(np.mean(list(notas.values())), 2)
            estudiante = {
                'nombre': f'Estudiante_{i}',
                'asistencia': 100,
                **notas,
                'promedio': promedio,
                'estado': 'Desaprobado'
            }
        data.append(estudiante)

    df = pd.DataFrame(data)
    df.to_csv(CSV_FILE, index=False)
    return df

def preparar_datos():
    if not os.path.exists(CSV_FILE):
        df = generar_datos()
    else:
        df = pd.read_csv(CSV_FILE)

    df.rename(columns={
        'ciencia y tecnologia': 'ciencia_y_tecnologia',
        'personal social': 'personal_social'
    }, inplace=True)

    X = df[['asistencia', 'matematica', 'comunicacion', 'ciencia_y_tecnologia', 'personal_social', 'promedio']]
    y = df['estado']
    return X, y, df

def entrenar_y_guardar_modelo():
    X, y, _ = preparar_datos()
    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X, y)
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(modelo, f)

def cargar_y_predecir(datos):
    if not os.path.exists(MODEL_FILE):
        entrenar_y_guardar_modelo()
    with open(MODEL_FILE, "rb") as f:
        modelo = pickle.load(f)
    df = pd.DataFrame([datos])
    df = df[['asistencia', 'matematica', 'comunicacion', 'ciencia_y_tecnologia', 'personal_social', 'promedio']]
    return modelo.predict(df)[0]

def guardar_estudiante(data):
    df = pd.read_csv(CSV_FILE) if os.path.exists(CSV_FILE) else pd.DataFrame()
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)

# ----------------------------------------
# VISUALIZACIONES
# ----------------------------------------

def generar_visualizaciones(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(data=df, x='estado')
    plt.title('Distribución de Estados')
    plt.savefig(f'{IMG_DIR}/estado_distribucion.png')
    plt.close()

    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df, x='estado', y='promedio')
    plt.title('Promedio por Estado')
    plt.savefig(f'{IMG_DIR}/promedio_estado.png')
    plt.close()

# ----------------------------------------
# PROCESAMIENTO DE FORMULARIO
# ----------------------------------------

def procesar_datos_formulario(form):
    datos = form.to_dict()
    if datos.get("asistencia") == "Ausente":
        datos.update({
            'asistencia': 0,
            'matematica': 0.0,
            'comunicacion': 0.0,
            'ciencia_y_tecnologia': 0.0,
            'personal_social': 0.0,
            'promedio': 0.0
        })
    else:
        datos['asistencia'] = 100
        for campo in ['matematica', 'comunicacion', 'ciencia_y_tecnologia', 'personal_social']:
            try:
                datos[campo] = float(datos.get(campo, 0.0))
            except ValueError:
                datos[campo] = 0.0
        # Calcular promedio del lado del servidor
        notas = [datos['matematica'], datos['comunicacion'], datos['ciencia_y_tecnologia'], datos['personal_social']]
        datos['promedio'] = round(sum(notas) / len(notas), 2)
    return datos

# ----------------------------------------
# FLASK ROUTES
# ----------------------------------------

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/registro", methods=["GET", "POST"])
def registro():
    if request.method == "POST":
        datos = procesar_datos_formulario(request.form)
        estado = cargar_y_predecir(datos)
        datos_final = {'nombre': datos['nombre'], 'estado': estado, **datos}
        guardar_estudiante(datos_final)
        return redirect(url_for("registro"))
    return render_template("registro.html")

@app.route("/guardar", methods=["POST"])
def guardar():
    datos = procesar_datos_formulario(request.form)
    estado = datos.get("estado", "Aprobado")
    datos['estado'] = estado
    guardar_estudiante(datos)
    return redirect(url_for("registro"))

@app.route("/generar_datos")
def generar():
    generar_datos()
    return redirect(url_for("ver_datos"))

@app.route("/ver_datos")
def ver_datos():
    df = pd.read_csv(CSV_FILE)
    return render_template("ver_datos.html", tablas=[df.to_html(classes='table table-striped')])

@app.route("/entrenar_modelos")
def entrenar():
    entrenar_y_guardar_modelo()
    _, _, df = preparar_datos()
    generar_visualizaciones(df)
    return redirect(url_for("resultados"))

@app.route("/resultados")
def resultados():
    return render_template("resultados.html")

@app.route("/prediccion", methods=["GET", "POST"])
def prediccion():
    resultado = None
    if request.method == "POST":
        datos = procesar_datos_formulario(request.form)
        resultado = cargar_y_predecir(datos)
    return render_template("prediccion.html", prediccion=resultado)

@app.route("/estadisticas")
def estadisticas():
    if not os.path.exists(CSV_FILE):
        return render_template("estadisticas.html", resumen={})
    df = pd.read_csv(CSV_FILE)
    resumen = df.groupby('estado').agg(['mean', 'std']).round(2).to_dict()
    return render_template("estadisticas.html", resumen=resumen)

if __name__ == "__main__":
    app.run(debug=True)