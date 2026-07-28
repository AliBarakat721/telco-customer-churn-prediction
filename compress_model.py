import joblib

model = joblib.load("models/best_model.pkl")

joblib.dump(
    model,
    "models/best_model_compressed.pkl",
    compress=9
)

print("Model compressed successfully")
