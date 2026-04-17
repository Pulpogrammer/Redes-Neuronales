import random

#agrego  sesgo a cada lista interna(ejemplos), se retorna la lista modificada
def agregar_sesgo(datos_entrada):
    for i in datos_entrada:
        i.append(1)
    return datos_entrada    

#calculo el producto interno
def exitacion(d_actual, w):
    resultado = 0
    for i in range(len(d_actual)):
        resultado += d_actual[i] * w[i]
    return resultado


def signo (h):
    return 1 if h > 0 else -1

#deltas para cada cavor del dato actual
def calc_deltas(tasa_aprendizaje, e, d_actual):
    deltas = []
         
    for i in range(len(d_actual)):
        ajuste = tasa_aprendizaje * e * d_actual[i]
        deltas.append(ajuste)
    return deltas    

#actualizo los valores de cada w actual
def actualizar_w(w, deltas_w):
    for i in range(len(deltas_w)):
        w[i]+= deltas_w[i]
    return w    
    
# calculamos el error promedio para cada ejemplos   
def calc_error_global(datos_entrada, y_deseadas, w):
    suma_errores = 0.0
    for i in range(len(datos_entrada)):
        d_actual = datos_entrada[i]
        y_actual = y_deseadas[i]
        O = calcular_salida(d_actual, w)
        
        e = (y_actual - O)**2
        suma_errores += e 
    return suma_errores/ 2
            
# tanto la exitacion y la salida se utilizan en el calculo principal del dato random como en el error global
# para no repetir codigo se utiliza esta funcion que devuelve la salida
def calcular_salida(d_actual, w):
    h = exitacion(d_actual,w)
    salida = signo(h)
    return salida  
            
            
def perceptron_lineal(datos_entrada,y_deseadas,cota, tasa_aprendizaje):
    print(f"Iniciando entrenamiento | Ejemplos: {len(datos_entrada)} | Cota: {cota} | Eta: {tasa_aprendizaje}")

    #agregamos el sesgo
    datos_entrada = agregar_sesgo(datos_entrada)
    
    #cantidad de pesos, lo obtengo haciando referencia a la sublista 0
    n_pesos = len(datos_entrada[0])
    
    #creo los pesos dependiendo del tamaño de la sublista 
    w = [0.0] * n_pesos
    
    print(f"Pesos iniciales: {w}")
    i= 0
    error = 1
    error_min = len(datos_entrada) * 2
    
    w_min = None
    
    #puede deternerse en 2 casos:
        #por que el error es menor al limite o por la cantidad de iteraciones de cota
    while error> 0 and i < cota:
        print(f"\n--- Iteración {i} ---")
        i_random = random.randrange(len(datos_entrada))
        
        d_actual = datos_entrada[i_random]
        y_actual = y_deseadas[i_random]
        print(f"Ejemplo elegido: {d_actual} | Y deseada: {y_actual}")
        
        #le asigno la funcion para calcular el producto interno y el signo
        O = calcular_salida(d_actual,w) 
        e = y_actual - O
        print(f"Error puntual e: {e}")
        
        deltas_w = calc_deltas(tasa_aprendizaje, e, d_actual)
        print(f"Deltas: {deltas_w}")
        
        
        w = actualizar_w(w, deltas_w)
        print(f"Pesos actualizados: {w}")
        
        error = calc_error_global(datos_entrada, y_deseadas, w)
        print(f"Error global: {error:.4f}")


        if error < error_min:
            error_min = error
            w_min=w.copy()
            print(f">>> Nuevo mínimo: {error_min:.4f} | w_min guardado: {w_min}")
               
        i+=1
    print(f"\n=== Entrenamiento finalizado ===")
    print(f"Iteraciones realizadas: {i}")
    print(f"Error mínimo alcanzado: {error_min:.4f}")
    print(f"Mejores pesos: {w_min}")
    return  w_min     
    


datos_entrada = [[-1, 1],[1, -1],[-1, -1],[1, 1]]

# valores objetivo
y_deseadas = [-1,-1, -1, 1]
#n de iteraciones
cota = 200
#eta
tasa_aprendizaje= 0.01
perceptron_lineal(datos_entrada, y_deseadas, cota, tasa_aprendizaje)
