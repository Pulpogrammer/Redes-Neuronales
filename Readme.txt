# Perceptron Simple

Implementación de un perceptron simple en Python, basada en el pseudocódigo clásico del algoritmo de aprendizaje por corrección de error.

Es mi primer proyecto de Inteligencia Artificial, desarrollado con el objetivo de entender los fundamentos del aprendizaje automático.

## ¿Qué hace?

Dado un conjunto de ejemplos con sus etiquetas deseadas (`1` o `-1`), el perceptron ajusta sus pesos iterativamente hasta clasificarlos correctamente o alcanzar el límite de iteraciones.

## ¿Cómo ejecutarlo?

No requiere librerías externas, solo Python 3.

```bash
python perceptron.py
```

## Ejemplo de uso

```python
datos_entrada = [[0.1, 0.2], [3.2, 1.2], [0.2, 2.1], [5.2, 1.1]]
y_deseadas = [1, 1, -1, -1]
cota = 200

w_min = perceptron_lineal(datos_entrada, y_deseadas, cota)
```

## Conceptos aplicados

- **Exitación**: producto interno entre el ejemplo y los pesos
- **Activación**: función signo aplicada sobre la exitación
- **Error global**: suma de errores cuadráticos sobre todos los ejemplos
- **w_min**: se guarda el mejor conjunto de pesos encontrado durante el entrenamiento
- **Aprendizaje **: en cada iteración se elige un ejemplo al azar
