import pandas as pd
import json
from datetime import datetime

def preprocess_data():
    # 1. Cargar datos crudos
    df = pd.read_csv('data/raw/abalone_raw.csv')
    
    # 2. Calcular nueva columna 'age'
    df['age'] = df['rings'] + 1.5
    
    # 3. Codificación de la variable categórica 'sex'
    df = pd.get_dummies(df, columns=['sex'], prefix='sex')
    
    # 4. Revisión de valores atípicos (Criterio: altura mayor a 0.4 o igual a 0)
    outliers_mascara = (df['height'] > 0.4) | (df['height'] == 0)
    df_limpio = df[~outliers_mascara]
    conteo = outliers_mascara.sum()
    
    # 5. Guardar datos procesados
    df_limpio.to_csv('data/processed/abalone_clean.csv', index=False)
    
    # 6. Generar log de transformaciones
    log_data = {
        "version": "v1",
        "created_at": datetime.now().strftime("%Y-%m-%d"),
        "source": "data/raw/abalone_raw.csv",
        "transformations": [
            "added derived feature 'age = rings + 1.5'",
            "encoded sex with one-hot encoding (sex_F, sex_I, sex_M)",
            f"removed {conteo} outliers with height > 0.4 or height == 0"
        ]
    }
    
    with open('data/data_log.json', 'w') as f:
        json.dump(log_data, f, indent=4)
        
if __name__ == "__main__":
    preprocess_data()