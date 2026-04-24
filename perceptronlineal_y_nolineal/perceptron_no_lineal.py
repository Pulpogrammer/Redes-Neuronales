import random
from math import tanh
import requests
import numpy as np
from io import StringIO

class PerceptronNoLineal:
    
    def __init__(self, datos_entrada, y_deseadas, COTA, tasa_aprendizaje, beta):
        self._datos_entrada = datos_entrada
        self._y_deseadas = y_deseadas
        self._COTA = COTA
        self._tasa_aprendizaje = tasa_aprendizaje
        self._beta = beta
 
    #agrego  sesgo a cada lista interna(ejemplos), se retorna una nueva lista modificada
    def agregar_sesgo(self):
        return [i + [1] for i in self._datos_entrada]

    #calculo el producto intero
    def exitacion(self, d_actual, w):
        resultado = 0
        for i in range(len(d_actual)):
            resultado += d_actual[i] * w[i]
        return resultado


    #deltas para cada valor del dato actual
    def calc_deltas(self, e, d_actual, O):
        deltas = []
        derivada = self._beta * (1-O**2)
        for i in range(len(d_actual)):
            ajuste = self._tasa_aprendizaje * e * derivada * d_actual[i]
            deltas.append(ajuste)
        return deltas

    #actualizo los valores de cada w actual
    def actualizar_w(self, w, deltas_w):
        for i in range(len(deltas_w)):
            w[i]+= deltas_w[i]
        return w    
        
    # calculamos el error promedio para cada ejemplos   
    def calc_error_global(self, datos_entrada, w):
        suma_errores = 0.0
        for i in range(len(datos_entrada)):
            d_actual = datos_entrada[i]
            y_actual = self._y_deseadas[i]
            O = self.calcular_salida(d_actual, w)
            
            e = (y_actual - O)**2
            suma_errores += e
        return suma_errores / 2
                
    # tanto la exitacion y la salida se utilizan en el calculo principal del dato random como en el error global
    # para no repetir codigo se utiliza esta funcion que devuelve la salida
    def calcular_salida(self, d_actual, w):
        h = self.exitacion(d_actual,w)
        return tanh(self._beta * h)
                                
    def iniciar_entrenamiento(self):
        print(f"iniciando entrenamiento | ejemplos: {len(self._datos_entrada)} | COTA: {self._COTA} | eta: {self._tasa_aprendizaje} | Objetivos: {len(self._y_deseadas)}" )
 
        #agrego sesgo y retorno una nueva lista
        datos_entrada = self.agregar_sesgo()
        
        #cantidad de pesos, lo obtengo haciando referencia a la sublista 0
        n_pesos = len(datos_entrada[0])
        
        #creo los pesos dependiendo del tamaño de la sublista 
        w = [0.0] * n_pesos
        
        print(f"pesos inicial: {w}")
        i= 0
        error = 1
        error_min = len(datos_entrada) * 2
        
        w_min = None
        
        while error> 0 and i < self._COTA:
            
            i_random = random.randrange(len(datos_entrada))
            
            d_actual = datos_entrada[i_random]
            y_actual = self._y_deseadas[i_random]
            #le asigno la funcion para calcular la exitacion y tanh 
            O = self.calcular_salida(d_actual,w) 
            #calculo el error para este dato
            e = y_actual - O
           
#obtengo los deltas para luego actualizar w
            deltas_w = self.calc_deltas(e, d_actual, O)

            
            #actualizo w
            w = self.actualizar_w(w, deltas_w)

            error = self.calc_error_global(datos_entrada, w)
            print(f"error global: {error:.4f}")


            if error < error_min:
                error_min = error
                w_min=w.copy()

            i+=1
        print(f"\n=== entrenamiento finalizado ===")
        print(f"iteraciones realizadas: {i}")
        return  w_min, error_min

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




datos_entrada = cargar_entrada()
y_deseadas = cargar_salida()


# normalizo salidas entre intervalo [-1,1] para la tangente
y_min = min(y_deseadas)
y_max = max(y_deseadas)
y_normalizadas = []
for y in y_deseadas:
    y_normalizadas.append(normalizar_salida(y, y_min, y_max))

#n de iteraciones
COTA = 400
#eta
tasa_aprendizaje = 0.1
beta = 0.8

perceptron = PerceptronNoLineal(datos_entrada, y_normalizadas, COTA, tasa_aprendizaje, beta)
w_min, error_min = perceptron.iniciar_entrenamiento()

print(f"mejores pesos: {w_min}")
print(f"error minimo alcanzado: {error_min:.4f}")

# evaluo 5 ejemplos con w_min
print(f"rvaluacion con w_min")
datos_con_sesgo = [fila + [1] for fila in datos_entrada]
indice = datos_con_sesgo[:5]

for i in range(len(indice)):
    objetivo   = y_normalizadas[i]
    prediccion = perceptron.calcular_salida(datos_con_sesgo[i], w_min)
    print(f"ejemplo {i + 1} | objetivo: {objetivo} | prediccion: {prediccion}")


#para el punto b del 2, en donde hay que tomar datos de entrenamiento y prueba, elegí el 80/20

# tomo el total de datos de entrada y me quedo con el 80%
cantidad_entrenamiento = int(len(datos_entrada) * 0.8)

# tomo los datos entrada del 80% hacia atras para entrenamiento 
X_entrenamiento = datos_entrada[:cantidad_entrenamiento]

#tomo los Y objetivo del 80% hacia  atras para entrenamiento
y_entrenamiento = y_normalizadas[:cantidad_entrenamiento]

#tomo los datos entrada para testeo, que sería el 20% restante
X_testeo  = datos_entrada[cantidad_entrenamiento:]

#tomo los Y objetivo para testeo, que sería el 20% restante
y_testeo  = y_normalizadas[cantidad_entrenamiento:]

print(f"\n ejemplos entrenamiento: {len(X_entrenamiento)} | Ejemplos test: {len(X_testeo)}")

# entreno los datos seleccionados
perceptron = PerceptronNoLineal(X_entrenamiento, y_entrenamiento, COTA, tasa_aprendizaje, beta)

w_min, error_min = perceptron.iniciar_entrenamiento()

print(f"Mejores pesos: {w_min}")
print(f"Error minimo alcanzado: {error_min:.4f}")

# agrego sesgo a los datos del test entrada
X_testeo_sesgo = [fila + [1] for fila in X_testeo]


print(f"\n inicio de testeo")

#testeo probando el w_min que encontré
for i in range(len(X_testeo_sesgo)):
    objetivo   = y_testeo[i]
    prediccion = perceptron.calcular_salida(X_testeo_sesgo[i], w_min)
    print(f"ejemplo {i + 1}, objetivo: {objetivo}, predicción: {prediccion}")


# ¿Como podría escojer el mejor conjunto de entrenamiento?
# Para elegir el mejor conjunto de entrenamiento se podria utilizar validacion cruzada, probar diferentes combinaciones.
# Dibido el total de datos en k partes iguales, en cada prueba se entrena con k-1 partes y se testea con la particion con la 
# parte restante que no vio, rotando cual queda afuera en cada prueba. Al finalizar las k pruebas se comparan los errores de test
# y el conjunto de entrenamiento que produjo el menor error sobre datos que no vio es el mejor, ya que demuestra que el modelo aprendio
# el patron general y no solo memorizo los datos de entrenamiento. A mayor k mas confiable la eleccion pero mayor costo computacional.

# Ejemplo: elijo a k = 10, divido en 10 partes con 20 ejemplos cada uno, de los 10 partes tomo 9 en cada prueba para entrenamiento dejando una parte para testeo
#  en cada prueba la parte que se testea sera diferente, es decir, las 10 partes pasaran por testeo, 1 en cada prueba. Al finaliizar en cada testeo observo el conjunto que me haya dado el menor error,
# el cual seria el mejor conjunto de entrenamiento

    
