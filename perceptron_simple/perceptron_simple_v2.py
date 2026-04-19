import random

class PerceptronSimple:
    
    def __init__(self, datos_entrada, y_deseadas, COTA, tasa_aprendizaje):
        self._datos_entrada = datos_entrada
        self._y_deseadas = y_deseadas
        self._COTA = COTA
        self._tasa_aprendizaje = tasa_aprendizaje

    #agrego  sesgo a cada lista interna(ejemplos), se retorna una nueva lista modificada
    def agregar_sesgo(self):
        return [i + [1] for i in self._datos_entrada]

    #calculo el producto intero
    def exitacion(self, d_actual, w):
        resultado = 0
        for i in range(len(d_actual)):
            resultado += d_actual[i] * w[i]
        return resultado

    def signo (self, h):
        return 1 if h >= 0 else -1

    #deltas para cada valor del dato actual
    def calc_deltas(self, e, d_actual):
        deltas = []
            
        for i in range(len(d_actual)):
            ajuste = self._tasa_aprendizaje * e * d_actual[i]
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
        salida = self.signo(h)
        return salida  
                                
    def iniciar_entrenamiento(self):
        print(f"Iniciando entrenamiento | Ejemplos: {len(self._datos_entrada)} | COTA: {self._COTA} | Eta: {self._tasa_aprendizaje} | Objeticos: {self._y_deseadas}" )
 
        #agrego sesgo y retorno una nueva lista
        datos_entrada = self.agregar_sesgo()
        
        #cantidad de pesos, lo obtengo haciando referencia a la sublista 0
        n_pesos = len(datos_entrada[0])
        
        #creo los pesos dependiendo del tamaño de la sublista 
        w = [0.0] * n_pesos
        
        print(f"Pesos inicial: {w}")
        i= 0
        error = 1
        error_min = len(datos_entrada) * 2
        
        w_min = None
        
        while error> 0 and i < self._COTA:
            
            i_random = random.randrange(len(datos_entrada))
            
            d_actual = datos_entrada[i_random]
            y_actual = self._y_deseadas[i_random]
            #le asigno la funcion para calcular el producto interno y el signo
            O = self.calcular_salida(d_actual,w) 
            e = y_actual - O
            print(f"Error puntual e: {e}")
            
            deltas_w = self.calc_deltas(e, d_actual)
            print(f"Deltas: {deltas_w}")
        
            w = self.actualizar_w(w, deltas_w)
            print(f"Pesos actualizados: {w}")
            
            error = self.calc_error_global(datos_entrada, w)
            print(f"Error global: {error:.4f}")
            if error < error_min:
                error_min = error
                w_min=w.copy()
                print(f">>> Nuevo mínimo: {error_min:.4f} | w_min guardado: {w_min}")
                
            i+=1
        print(f"\n=== Entrenamiento finalizado ===")
        print(f"Iteraciones realizadas: {i}")
        return  w_min, error_min
    

datos_entrada = [[-1, 1],[1, -1],[-1, -1],[1, 1]]
# y deseadas, agregue los tipos dentro de un diccionario para verificar en cada caso
y_deseadas={"AND": [-1,-1, -1, 1],
            "XOR": [1, 1, -1, -1] }

#n de iteraciones
COTA = 200
#eta
tasa_aprendizaje= 0.01
perceptron = PerceptronSimple(datos_entrada, y_deseadas.get("XOR"), COTA, tasa_aprendizaje)
w_min, error_min = perceptron.iniciar_entrenamiento()
print(f"Mejores pesos: {w_min}")
print(f"Error mínimo alcanzado: {error_min:.4f}")
    
