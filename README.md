# Predicción de edad en Abalones (Proyecto de Reproducibilidad)

## Descripción
Este proyecto tiene como objetivo construir un modelo de Machine Learning simple para predecir la edad de los moluscos (Abalone) basándose en mediciones físicas. El enfoque principal está en aplicar buenas prácticas de reproducibilidad, gestión de entornos, control de versiones y registro de experimentos.

## Requisitos y Entorno
El proyecto fue desarrollado utilizando **Python 3.15.1** 

### Instrucciones para reproducir el entorno:
1. Clonar el repositorio.
2. Crear un entorno virtual:
   `python -m venv venv`
3. Activar el entorno:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
4. Instalar las dependencias exactas:
   `pip install -r requirements.txt`


## Estructura del Proyecto
- `data/`: Contiene los datos crudos, los procesados y el registro de transformaciones.
- `src/`: Scripts de código fuente para preprocesamiento y entrenamiento.
- `results/`: Métricas y registros de los experimentos realizados.
- `models/`: Modelos entrenados serializados.


## Opcional
Si deseas volver a descargar los datos originales desde cero, puedes ejecutar `python src/download_data.py`


### Preprocesamiento de Datos
Para generar el dataset limpio y registrar las transformaciones, ejecutar desde la carpeta `src/`:
`python preprocess.py`
Esto tomará los datos de `data/raw/abalone_raw.csv` y generará `data/processed/abalone_clean.csv` junto con `data/data_log.json`.