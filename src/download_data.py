import pandas as pd

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"

# Usamos minúsculas para evitar errores en el preprocesamiento
column_names = ["sex", "length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight", "rings"]

print("Descargando datos...")
df = pd.read_csv(url, names=column_names)

# Guardamos el archivo CSV en la carpeta data/raw sin la columna de índices
df.to_csv("data/raw/abalone_raw.csv", index=False)
print("¡Datos guardados exitosamente en data/raw/abalone_raw.csv!")
