import random

#agrego sesgo a cada lista interna(ejemplos), se retorna una nueva lista modificada
def agregar_sesgo(datos_entrada):
    return [i + [1] for i in datos_entrada]

#calculo el producto interno
def exitacion(d_actual, w):
    resultado = 0
    for i in range(len(d_actual)):
        resultado += d_actual[i] * w[i]
    return resultado

#funcion de activacion
def signo(h):
    return 1 if h >= 0 else -1

#deltas para el w actual
def calc_deltas(tasa_aprendizaje, e, d_actual):
    deltas = []
    for i in range(len(d_actual)):
        ajuste = tasa_aprendizaje * e * d_actual[i]
        deltas.append(ajuste)
    return deltas

#actualizo los valores del w actual
def actualizar_w(w, deltas_w):
    for i in range(len(deltas_w)):
        w[i] += deltas_w[i]
    return w

# calculo la cantidad de errores con el w actualizado
def calc_error_global(datos_entrada, y_deseadas, w):
    suma_errores = 0
    for i in range(len(datos_entrada)):
        d_actual = datos_entrada[i]
        y_actual = y_deseadas[i]
        O = calcular_salida(d_actual, w)
        if y_actual != O:
            suma_errores += 1
    return suma_errores

# como la exitacion y la salida se utilizan para la entrada random y el conteo de errores
# cree esta funcion para no repetir código
def calcular_salida(d_actual, w):
    h = exitacion(d_actual, w)
    salida = signo(h)
    return salida


def perceptron_simple(datos_entrada, y_deseadas, cota, tasa_aprendizaje):
    print(f"Iniciando entrenamiento | Ejemplos: {len(datos_entrada)} | COTA: {cota} | Eta: {tasa_aprendizaje} | Objetivos: {y_deseadas}")

    #agrego sesgo y retorno una nueva lista
    datos_entrada = agregar_sesgo(datos_entrada)

    #cantidad de pesos, lo obtengo haciendo referencia a la sublista 0
    n_pesos = len(datos_entrada[0])

    #creo los pesos dependiendo del tamaño de la sublista
    w = [0.0] * n_pesos

    print(f"Pesos inicial: {w}")
    i = 0
    error = 1
    error_min = len(datos_entrada) * 2

    w_min = None

    while error > 0 and i < cota:
        i_random = random.randrange(len(datos_entrada))

        d_actual = datos_entrada[i_random]
        y_actual = y_deseadas[i_random]

        #le asigno la funcion para calcular el producto interno y el signo
        O = calcular_salida(d_actual, w)
        e = y_actual - O

        #lista deltas para cada w
        deltas_w = calc_deltas(tasa_aprendizaje, e, d_actual)

        w = actualizar_w(w, deltas_w)

        #verifico la cantidad de error global
        error = calc_error_global(datos_entrada, y_deseadas, w)
        print(f"Error global: {error:.4f}")

        #si el error es menor al ya guardado actualizo
        if error < error_min:
            error_min = error
            w_min = w.copy()
            print(f">>> Nuevo mínimo: {error_min:.4f} | w_min guardado: {w_min}")

        i += 1

    print(f"\n=== Entrenamiento finalizado ===")
    print(f"Iteraciones realizadas: {i}")
    return w_min, error_min


# x datos entrada
datos_entrada = [[-1, 1],[1, -1],[-1, -1],[1, 1]]

# y deseadas
y_deseadas = {"AND": [-1, -1, -1, 1],
              "XOR": [1, 1, -1, -1]}

#n de iteraciones
COTA = 200
#eta
tasa_aprendizaje = 0.01

w_min, error_min = perceptron_simple(datos_entrada, y_deseadas.get("AND"), COTA, tasa_aprendizaje)
print(f"Mejores pesos: {w_min}")
print(f"Error mínimo alcanzado: {error_min:.4f}")