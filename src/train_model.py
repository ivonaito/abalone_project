import pandas as pd
import json
import os
import csv
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def train_model():
    # Asegurar que las carpetas existan
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # Cargar los datos limpios
    df = pd.read_csv('data/processed/abalone_clean.csv')

    # Separar características (X) y objetivo (y)
    # Eliminamos 'rings' para evitar data leakage
    X = df.drop(columns=['age', 'rings'])
    y = df['age']

    # Dividir en train y test 
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

    # Crear el Pipeline y entrenar
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=SEED))
    ])
    pipe.fit(X_train, y_train)

    # Evaluar el modelo
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Guardar el modelo comprimido
    joblib.dump(pipe, 'models/model_v1.pkl', compress=3)

    # 7. Guardar métricas e hiperparámetros en JSON
    metrics_data = {
        "model_version": "v1",
        "seed": SEED,
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 10,
            "algorithm": "RandomForestRegressor"
        },
        "metrics": {
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4)
        }
    }
    with open('results/metrics_v1.json', 'w') as f:
        json.dump(metrics_data, f, indent=4)

    # Registrar ejecución en el log CSV
    exp_log_path = 'results/experiments_log.csv'
    file_exists = os.path.isfile(exp_log_path)
    
    with open(exp_log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['experiment_id', 'date_time', 'model_version', 'hyperparameters', 'mae', 'rmse', 'r2'])
        
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        date_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        writer.writerow([
            exp_id, date_time, "v1", 
            str(metrics_data['hyperparameters']), 
            round(mae, 4), round(rmse, 4), round(r2, 4)
        ])
        
    print("¡Modelo entrenado y registrado exitosamente!")

if __name__ == "__main__":
    train_model()