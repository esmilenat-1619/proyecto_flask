from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# cargar datos
data = load_breast_cancer()

# ⚠️ usar solo 5 columnas para evitar errores después
X = data.data[:, :5]
y = data.target

# dividir datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# crear modelo
model = LogisticRegression(max_iter=5000)

# entrenar
model.fit(X_train, y_train)

# guardar modelo
joblib.dump(model, "model.pkl")

print("✅ Modelo guardado correctamente como model.pkl")