# Implementación del análisis de dataset

Se actualizó `ml_pipeline.py` para tomar como referencia directa el notebook de clase de MLP tuning.

## Estado de implementación vs notebook

- **(ya estaba implementado, coincide con notebook)** uso de `Pipeline` para integrar preprocesamiento + modelo.
- **(extraido del notebook de clase)** modelo base de clasificación con `MLPClassifier`.
- **(extraido del notebook de clase)** validación cruzada con `cross_val_score`.
- **(extraido del notebook de clase)** inspección de combinaciones con `ParameterGrid`.
- **(extraido del notebook de clase)** búsqueda exhaustiva con `GridSearchCV`.
- **(extraido del notebook de clase)** búsqueda aleatoria con `RandomizedSearchCV`.
- **(extraido del notebook de clase)** comparación final entre Base vs GridSearchCV vs RandomizedSearchCV en validación y test.
- **(extraido del notebook de clase)** flujo equivalente para regresión con `MLPRegressor` y scoring `r2`.

## Cómo ejecutar

```bash
pip install numpy pandas scikit-learn
python ml_pipeline.py
```

## Nota

El script toma la última columna del CSV como variable objetivo. Si necesitas fijar otra columna objetivo, se puede parametrizar por CLI.
