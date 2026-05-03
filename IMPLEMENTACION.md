# Implementación del análisis de dataset

Este repositorio ahora incluye una implementación base en `ml_pipeline.py` que sigue el plan propuesto:

1. Carga del dataset `city_day.csv`.
2. Detección automática de tipo de tarea (clasificación/regresión).
3. Separación `train/test` con `random_state` fijo para reproducibilidad.
4. Pipeline de preprocesamiento con:
   - imputación,
   - escalado para variables numéricas,
   - one-hot encoding para categóricas.
5. Validación cruzada (`KFold` o `StratifiedKFold`).
6. Optimización de hiperparámetros:
   - `GridSearchCV` en clasificación,
   - `RandomizedSearchCV` en regresión.
7. Evaluación final sobre conjunto de test no visto.

## Cómo ejecutar

```bash
pip install numpy pandas scikit-learn
python ml_pipeline.py
```

## Nota

El script toma la última columna del CSV como variable objetivo. Si necesitas una columna específica, podemos parametrizarlo con argumentos CLI.
