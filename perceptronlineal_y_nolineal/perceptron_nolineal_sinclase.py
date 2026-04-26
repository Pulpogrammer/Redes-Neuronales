import random
from math import tanh
import requests
import numpy as np
from io import StringIO


# agrego sesgo a cada lista interna (ejemplos), se retorna una nueva lista modificada
def agregar_sesgo(datos_entrada):
    return [i + [1] for i in datos_entrada]


# calculo el producto interno
def exitacion(d_actual, w):
    resultado = 0
    for i in range(len(d_actual)):
        resultado += d_actual[i] * w[i]
    return resultado


# tanto la exitacion y la salida se utilizan en el calculo principal del dato random como en el error global
# para no repetir codigo se utiliza esta funcion que devuelve la salida
def calcular_salida(d_actual, w, beta):
    h = exitacion(d_actual, w)
    return tanh(beta * h)


# deltas para cada valor del dato actual
def calc_deltas(e, d_actual, O, tasa_aprendizaje, beta):
    deltas = []
    derivada = beta * (1 - O**2)
    for i in range(len(d_actual)):
        ajuste = tasa_aprendizaje * e * derivada * d_actual[i]
        deltas.append(ajuste)
    return deltas


# actualizo los valores de cada w actual
def actualizar_w(w, deltas_w):
    for i in range(len(deltas_w)):
        w[i] += deltas_w[i]
    return w


# calculamos el error global para todos los ejemplos
def calc_error_global(datos_entrada, y_deseadas, w, beta):
    suma_errores = 0.0
    for i in range(len(datos_entrada)):
        d_actual = datos_entrada[i]
        y_actual = y_deseadas[i]
        O = calcular_salida(d_actual, w, beta)
        e = (y_actual - O)**2
        suma_errores += e
    return suma_errores / 2


def iniciar_entrenamiento(datos_entrada, y_deseadas, COTA, tasa_aprendizaje, beta):
    print(f"iniciando entrenamiento | ejemplos: {len(datos_entrada)} | COTA: {COTA} | eta: {tasa_aprendizaje} | Objetivos: {len(y_deseadas)}")

    # agrego sesgo y retorno una nueva lista
    datos_con_sesgo = agregar_sesgo(datos_entrada)

    # cantidad de pesos, lo obtengo haciendo referencia a la sublista 0
    n_pesos = len(datos_con_sesgo[0])

    # creo los pesos dependiendo del tamaño de la sublista
    w = [0.0] * n_pesos

    print(f"pesos inicial: {w}")
    i = 0
    error = 1
    error_min = len(datos_con_sesgo) * 2
    w_min = None

    while error > 0 and i < COTA:
        i_random = random.randrange(len(datos_con_sesgo))
        d_actual = datos_con_sesgo[i_random]
        y_actual = y_deseadas[i_random]

        # calculo la exitacion y tanh
        O = calcular_salida(d_actual, w, beta)
        # calculo el error para este dato
        e = y_actual - O

        # obtengo los deltas para luego actualizar w
        deltas_w = calc_deltas(e, d_actual, O, tasa_aprendizaje, beta)

        # actualizo w
        w = actualizar_w(w, deltas_w)

        error = calc_error_global(datos_con_sesgo, y_deseadas, w, beta)
        print(f"error global: {error:.4f}")

        if error < error_min:
            error_min = error
            w_min = w.copy()

        i += 1

    print(f"\n=== entrenamiento finalizado ===")
    print(f"iteraciones realizadas: {i}")
    return w_min, error_min


# para y objetivo mayor a 1
def normalizar_salida(y, y_min, y_max):
    return ((y - y_min) / (y_max - y_min)) * 2 - 1


def cargar_entrada():
    url = "https://raw.githubusercontent.com/Pulpogrammer/Redes-Neuronales/refs/heads/main/perceptronlineal_y_nolineal/conjunto_entrenamiento.txt"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Error al descargar entrada: {resp.status_code}")
    matriz = np.loadtxt(StringIO(resp.text))
    if matriz.ndim == 1:
        return [matriz.tolist()]
    return [fila.tolist() for fila in matriz]


def cargar_salida():
    url = "https://raw.githubusercontent.com/Pulpogrammer/Redes-Neuronales/refs/heads/main/perceptronlineal_y_nolineal/salida_deseada.txt"
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Error al descargar salida: {resp.status_code}")
    array = np.loadtxt(StringIO(resp.text))
    if array.ndim == 0:
        return [float(array)]
    return array.tolist()


# ---- MAIN ----

datos_entrada = cargar_entrada()
y_deseadas = cargar_salida()

# normalizo salidas entre intervalo [-1,1] para la tangente
y_min = min(y_deseadas)
y_max = max(y_deseadas)
y_normalizadas = []
for y in y_deseadas:
    y_normalizadas.append(normalizar_salida(y, y_min, y_max))

# n de iteraciones
COTA = 400
# eta
tasa_aprendizaje = 0.1
beta = 0.8

w_min, error_min = iniciar_entrenamiento(datos_entrada, y_normalizadas, COTA, tasa_aprendizaje, beta)

print(f"mejores pesos: {w_min}")
print(f"error minimo alcanzado: {error_min:.4f}")

# evaluo 5 ejemplos con w_min
print(f"evaluacion con w_min")
datos_con_sesgo = agregar_sesgo(datos_entrada)
for i in range(5):
    objetivo   = y_normalizadas[i]
    prediccion = calcular_salida(datos_con_sesgo[i], w_min, beta)
    print(f"ejemplo {i + 1} | objetivo: {objetivo} | prediccion: {prediccion}")


# para el punto b, split 80/20
cantidad_entrenamiento = int(len(datos_entrada) * 0.8)

# tomo los datos entrada del 80% para entrenamiento
X_entrenamiento = datos_entrada[:cantidad_entrenamiento]

# tomo los Y objetivo del 80% para entrenamiento
y_entrenamiento = y_normalizadas[:cantidad_entrenamiento]

# tomo los datos entrada para testeo, el 20% restante
X_testeo = datos_entrada[cantidad_entrenamiento:]

# tomo los Y objetivo para testeo, el 20% restante
y_testeo = y_normalizadas[cantidad_entrenamiento:]

print(f"\nejemplos entrenamiento: {len(X_entrenamiento)} | Ejemplos test: {len(X_testeo)}")

w_min, error_min = iniciar_entrenamiento(X_entrenamiento, y_entrenamiento, COTA, tasa_aprendizaje, beta)

print(f"Mejores pesos: {w_min}")
print(f"Error minimo alcanzado: {error_min:.4f}")

# agrego sesgo a los datos del test
X_testeo_sesgo = agregar_sesgo(X_testeo)

print(f"\ninicio de testeo")

# testeo probando el w_min que encontré
for i in range(len(X_testeo_sesgo)):
    objetivo   = y_testeo[i]
    prediccion = calcular_salida(X_testeo_sesgo[i], w_min, beta)
    print(f"ejemplo {i + 1}, objetivo: {objetivo}, predicción: {prediccion}")


# ¿Como podría escoger el mejor conjunto de entrenamiento?
# Para elegir el mejor conjunto de entrenamiento se podria utilizar validacion cruzada, probar diferentes combinaciones.
# Divido el total de datos en k partes iguales, en cada prueba se entrena con k-1 partes y se testea con la
# parte restante que no vio, rotando cual queda afuera en cada prueba. Al finalizar las k pruebas se comparan los errores de test
# y el conjunto de entrenamiento que produjo el menor error sobre datos que no vio es el mejor, ya que demuestra que el modelo aprendio
# el patron general y no solo memorizo los datos de entrenamiento. A mayor k mas confiable la eleccion pero mayor costo computacional.

# Ejemplo: elijo a k = 10, divido en 10 partes con 20 ejemplos cada uno, de los 10 partes tomo 9 en cada prueba para entrenamiento dejando una parte para testeo
# en cada prueba la parte que se testea sera diferente, es decir, las 10 partes pasaran por testeo, 1 en cada prueba. Al finalizar en cada testeo observo el conjunto que me haya dado el menor error,
# el cual seria el mejor conjunto de entrenamiento