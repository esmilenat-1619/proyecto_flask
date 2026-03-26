from flask import Flask, request, render_template
import numpy as np
import joblib

app = Flask(__name__)

# cargar modelo
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    features = [float(x) for x in request.form.values()]
    final_features = [np.array(features)]
    prediction = model.predict(final_features)

    if prediction[0] == 0:
        resultado = "Maligno"
        color = "danger"
    else:
        resultado = "Benigno"
        color = "success"

    return render_template("index.html", prediction_text=resultado, color=color)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)


import joblib

joblib.dump(model, "model.pkl")