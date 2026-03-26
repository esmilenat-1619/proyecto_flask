from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Cargar modelo
model = joblib.load("model.pkl")  # ruta relativa al proyecto

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Aquí tu lógica de predicción
        data = request.form["input_name"]
        prediction = model.predict([[float(data)]])[0]
        return render_template("index.html", prediction=prediction)
    return render_template("index.html")