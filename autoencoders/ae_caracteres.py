import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
import numpy as np

# defino las etiquetas de cada font segun los caracteres que contiene cada bloque del archivo
# las necesito para poder mostrar que letra es cada punto cuando grafique el espacio latente
etiquetas_font1 = [' ','!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ':', ';', '<', '=', '>', '?']
etiquetas_font2 = ['@', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '[', '\\', ']', '^', '_']
etiquetas_font3 = ['`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', '{', '|', '}', '~', 'DEL']

# armo un diccionario para poder elegir las etiquetas correctas segun el font que use mas adelante
etiquetas_por_font = {
    "Font1": etiquetas_font1,
    "Font2": etiquetas_font2,
    "Font3": etiquetas_font3
}

# elijo con que font voy a trabajar en todo el script
# si despues quiero probar con otro bloque, solo cambio este string
font = "Font2"
etiquetas = etiquetas_por_font[font]


def cargar_datos(font, ruta_archivo='/content/fonts.txt'):
    # creo una lista vacia donde voy a ir guardando cada fila de 7 valores hexadecimales por caracter
    # cada elemento de esta lista va a representar un caracter completo, todavia sin convertir a binario
    patrones = []


    # uso esta bandera para saber cuando empezar a guardar las lineas del archivo
    # arranca en false porque todavia no llegue a la linea que marca el inicio del bloque que me interesa
    lineas_de_interes = False

    # abro el archivo de caracteres en modo lectura y especifico la codificacion utf-8
    # lo recorro linea por linea en vez de cargarlo todo en memoria de una vez
    with open(ruta_archivo, 'r', encoding='utf-8') as f:
        for linea in f:
            # busco la linea exacta donde arranca el bloque del font que recibi como parametro
            # por ejemplo si font vale "Font2" busco la aparicion de "Font2["
            if f"{font}[" in linea:
                # prendo la bandera para indicar que a partir de la proxima linea empiezo a guardar datos
                lineas_de_interes = True
                # salto a la siguiente iteracion sin procesar esta linea, porque es solo el encabezado del array
                continue

            # solo entro a este bloque si ya pase la linea de inicio del font que me interesa
            if lineas_de_interes:
                # corto la lectura del archivo en cuanto encuentro el cierre del array
                # esto evita que siga leyendo el font siguiente que viene a continuacion en el mismo archivo
                if "};" in linea:
                    break

                # me quedo solo con la parte de la linea que esta antes del comentario "//"
                # ademas le saco los espacios en blanco de los costados con strip
                contenido = linea.split("//")[0].strip()
                # saco las llaves de apertura y cierre, y las comas, porque no me aportan informacion util
                # despues de esto me queda solo una cadena con los numeros hexadecimales separados por espacios
                contenido = contenido.replace("{", "").replace("}", "").replace(",", "")

                # separo la cadena en una lista de strings, usando los espacios como separador
                partes = contenido.split()

                # cada caracter valido tiene exactamente 7 valores hexadecimales, uno por cada fila de pixeles
                # con esta condicion descarto lineas vacias o que no correspondan a la definicion de un caracter
                if len(partes) == 7:
                    # convierto cada string hexadecimal a un numero entero interpretandolo en base 16
                    filas_hex = [int(val, 16) for val in partes]
                    # agrego esta lista de 7 numeros a la lista general de patrones
                    patrones.append(filas_hex)

    # convierto la lista completa de patrones en un array de numpy
    # me queda con forma (32, 7): 32 caracteres, y 7 valores hexadecimales (ya en decimal) por cada uno
    datos_hex = np.array(patrones)

    # inicializo la matriz final donde van a quedar los pixeles ya expresados en binario
    # la lleno de ceros porque todavia no calcule ningun valor, es solo el molde vacio
    # uso float32 porque despues esta matriz va a entrar directo a la red neuronal, que trabaja con flotantes
    x_train = np.zeros((len(datos_hex), 7, 5), dtype=np.float32)

    # recorro cada uno de los caracteres que extraje del archivo
    for i in range(len(datos_hex)):
        # recorro las 7 filas de pixeles que componen ese caracter especifico
        for fila in range(7):
            # tomo el valor decimal correspondiente a esta fila de este caracter puntual
            valor_decimal = datos_hex[i, fila]
            # convierto ese numero a un string en formato binario, de exactamente 5 caracteres de largo
            # el formato 05b significa: representacion binaria (b), con un ancho minimo de 5 caracteres,
            # rellenando con ceros a la izquierda si el numero binario tiene menos de 5 cifras
            binario_str = f"{valor_decimal:05b}"
            # recorro cada caracter del string binario, que puede ser '0' o '1'
            # convierto cada uno a flotante y los guardo todos juntos en la posicion [i, fila] de x_train
            x_train[i, fila] = [float(bit) for bit in binario_str]

    # devuelvo la matriz final ya armada, lista para usarse como entrada del autocodificador
    return x_train


# funcion para contruir el modelo segun la arquitectura que le pase
def construir_modelo(arquitectura):
    # calculo cual es el indice del medio de la lista de arquitectura
    # ese va a ser el lugar donde pongo la capa que hace de espacio latente
    # uso division entera para que siempre me de un indice valido aunque la lista tenga longitud impar
    indice_latente = len(arquitectura) // 2

    # arranco un modelo secuencial vacio para ir agregando capas una despues de la otra
    model = Sequential()

    # defino que la entrada va a ser una matriz de 7x5, no un vector plano
    # esto es lo que mantiene la nocion de "dos dimensiones" que pide el enunciado
    model.add(Input(shape=(7, 5), name="entrada_2d"))

    # aplano la matriz de 7x5 a un vector de 35 valores
    # hago esto porque las capas Dense que vienen despues solo pueden procesar vectores, no matrices
    model.add(Flatten(name="aplanado"))

    # recorro la lista de arquitectura y voy agregando una capa Dense por cada numero de la lista
    # ese numero es la cantidad de neuronas que va a tener esa capa especifica
    for i, neuronas in enumerate(arquitectura):
        # si estoy en el indice del medio, le pongo el nombre espacio_latente
        # le pongo ese nombre puntual porque despues necesito ubicar esta capa para extraer sus valores
        if i == indice_latente:
            model.add(Dense(neuronas, activation='relu', name="espacio_latente"))
        else:
            # para el resto de las capas uso un nombre generico con el indice, solo para poder identificarlas
            model.add(Dense(neuronas, activation='relu', name=f"capa_{i}"))

    # agrego la capa final de reconstruccion con 35 neuronas, una por cada pixel original
    # uso sigmoid porque necesito que cada salida quede entre 0 y 1, ya que cada pixel es 0 o 1
    model.add(Dense(35, activation='sigmoid', name="reconstruccion_vector"))

    # vuelvo a darle forma de matriz 7x5 al vector de 35 que salio de la capa anterior
    # esto es lo que hace que la salida tambien respete las dos dimensiones, igual que la entrada
    model.add(Reshape((7, 5), name="salida_2d"))

    return model


# entrada VS reconstruccion de algunos caracteres
def mostrar_comparacion(model, x_entrada, etiquetas_mostrar,indices=[0, 1, 2, 3]):
    # le pido al modelo que prediga unicamente los caracteres en las posiciones que indique
    predicciones = model.predict(x_entrada[indices])

    # recorro cada indice que quiero comparar
    for idx, i in enumerate(indices):
        # squeeze me saca las dimensiones de tamaño 1 que sobran, asi me queda la matriz limpia de 7x5
        matriz_entrada = np.squeeze(x_entrada[i])
        matriz_recon = np.squeeze(predicciones[idx])

        #print(f"\n  Caracter: {etiquetas[i]}    ")
        print(f"\nCaracter: {etiquetas_mostrar[i]}")
        print("     ENTRADA      |     RECONSTRUIDA")
        print("------------------+-----------------")

        # recorro las 7 filas para armar el dibujo de cada una
        for fila in range(7):
            # si el pixel es mayor a 0.5 lo dibujo encendido, sino lo dejo como espacio en blanco
            # uso 0.5 como umbral porque es el punto medio entre 0 y 1, la frontera de decision mas natural
            entrada = "".join(["█" if p > 0.5 else " " for p in matriz_entrada[fila]])
            recon = "".join(["█" if p > 0.5 else " " for p in matriz_recon[fila]])

            print(f"  Fila {fila}:  {entrada:<10} |  {recon}")


# --- FUNCION PARA ENTRENAR Y EVALUAR UNA ARQUITECTURA DADA ---

def entrenar_evaluacion(x_train, arquitectura, lr=0.005, epochs=200, batch=4, prob_ruido=None):
    # construyo el modelo con la arquitectura que recibo como parametro
    model = construir_modelo(arquitectura)

    # configuro el optimizador adam, que adapta la tasa de aprendizaje a cada peso individualmente
    opt = keras.optimizers.Adam(learning_rate=lr)

    # compilo el modelo indicando la funcion de perdida y la metrica que quiero monitorear
    # uso binary_crossentropy porque mis datos son pixeles binarios (0 o 1), no valores continuos
    # mae es el error absoluto medio, lo agrego solo como metrica extra para tener mas informacion
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae'])

    # por defecto la entrada de entrenamiento es la misma x_train sin modificar
    x_entrada = x_train

    # si me pasaron una probabilidad de ruido, armo una version distorsionada de los datos
    if prob_ruido is not None:
       versiones_ruidosas = []
       versiones_limpias = []
       versiones_etiquetas = []

       # recorro todos los caracteres del dataset
       for indice in range(len(x_train)):
          caracter_original = x_train[indice]

          # genero 20 copias con ruido de este caracter especifico
          # aunque el porcentaje es el mismo, cada copia tiene pixeles distintos alterados
          # porque la mascara es aleatoria en cada iteracion
          for _ in range(20):
             copia = caracter_original.copy()
             mascara = np.random.rand(*copia.shape) < prob_ruido
             copia[mascara] = 1 - copia[mascara]
             versiones_ruidosas.append(copia)
             versiones_limpias.append(caracter_original)
             versiones_etiquetas.append(etiquetas[indice])

       # resultado: 32 caracteres x 20 copias = 640 ejemplos de entrenamiento
       x_entrada = np.array(versiones_ruidosas, dtype=np.float32)
       y_objetivo = np.array(versiones_limpias, dtype=np.float32)

       versiones_etiquetas = np.array(versiones_etiquetas)

       # Mezclamos los índices manteniendo la relación par entrada/objetivo
       indices_mezclados = np.random.permutation(len(x_entrada))
       x_entrada = x_entrada[indices_mezclados]
       y_objetivo = y_objetivo[indices_mezclados]
       versiones_etiquetas = versiones_etiquetas[indices_mezclados]
       etiquetas_mostrar = versiones_etiquetas
    else:
        y_objetivo = x_train
        etiquetas_mostrar = etiquetas

    # entreno el modelo. le paso x_entrada que puede ser array de 32 caracteres(7,5) o array de 20 caracteres (7,5) iguales con ruido

    # la salida y_objetivo  puede ser array de 32 caracteres(7,5) o array de 20 caracteres iguales limpios (7,5)
    historia = model.fit(
        x_entrada, y_objetivo,
        epochs=epochs,
        shuffle=True,
        validation_split=0.2,
        batch_size=batch,
        verbose=1
    )

    # me guardo el valor de loss de la ultima epoca para mostrarlo como referencia en el grafico
    ultimo_loss = historia.history['loss'][-1]

    # armo un string que describe la arquitectura para poner como titulo del grafico
    # por ejemplo si arquitectura es [24,12,4,12,24] me queda "35->24->12->4->12->24->35"
    arq_str = "35->" + "->".join(str(n) for n in arquitectura) + "->35"
    titulo = f"Loss | arch={arq_str} | lr={lr} | epochs={epochs} | batch={batch}"

    # grafico como bajo el loss a lo largo de las epocas, para poder analizar la convergencia
    plt.title(titulo)
    plt.plot(historia.history['loss'], label='Entrenamiento')
    plt.plot(historia.history['val_loss'], label='Validación')

    plt.ylabel('Funcion de perdida')
    plt.xlabel('Epocas')
    # agrego una linea horizontal punteada en el valor final, para ver de un vistazo donde se estabilizo
    plt.axhline(y=ultimo_loss, color='r', linestyle='--', label=f'Loss final: {ultimo_loss:.4f}')
    plt.legend(loc='upper right')
    plt.show()

    # muestro la comparacion visual de algunos caracteres ya entrenado el modelo
    #mostrar_comparacion(model, x_entrada)
    mostrar_comparacion(model, x_entrada, etiquetas_mostrar=etiquetas_mostrar)
    # devuelvo el modelo entrenado para poder seguir usandolo despues de llamar a esta funcion
    return model


# funcion para entrenar un modelo con espacio latente con 2 neuronas
def entrenar_espacio_latente(x_train, lr=0.001, epochs=500, batch=2):
    # armo manualmente esta arquitectura especifica porque necesito que la capa latente
    # tenga exactamente 2 neuronas, para poder representar cada caracter como un punto (x,y) en un plano
    model = Sequential()
    model.add(Input(shape=(7, 5), name="entrada_2d"))
    model.add(Flatten(name="aplanado"))
    model.add(Dense(24, activation='relu', name="capa_0"))
    # esta es la capa clave: con solo 2 neuronas, cada caracter queda resumido en 2 numeros
    # uso activacion lineal aca porque no quiero limitar el rango de salida como lo haria relu o sigmoid
    model.add(Dense(2, activation='linear', name="espacio_latente"))
    model.add(Dense(24, activation='relu', name="capa_1"))
    model.add(Dense(35, activation='sigmoid', name="reconstruccion_vector"))
    model.add(Reshape((7, 5), name="salida_2d"))

    # uso un learning rate mas bajo que en los otros casos porque con un cuello de botella tan angosto (2 neuronas)
    # la red es mas sensible a cambios bruscos en los pesos, y un lr alto podria hacerla oscilar
    opt = keras.optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['mae'])

    # entreno con mas epocas y batch mas chico, porque comprimir a solo 2 dimensiones es mas dificil
    # y necesito mas pasos de ajuste para que la red encuentre una buena representacion
    historia = model.fit(
        x_train, x_train,
        epochs=epochs,
        batch_size=batch,
        verbose=0
    )

    # grafico el loss de este entrenamiento particular
    ultimo_loss = historia.history['loss'][-1]
    plt.plot(historia.history['loss'])
    plt.title(f"Loss | arch=35->24->2->24->35 | lr={lr} | epochs={epochs} | batch={batch}")
    plt.axhline(y=ultimo_loss, color='r', linestyle='--', label=f'Loss final: {ultimo_loss:.4f}')
    plt.ylabel('Funcion de perdida')
    plt.xlabel('Epocas')
    plt.legend(loc='upper right')
    plt.show()

    # grafico los 32 caracteres proyectados en el espacio latente de 2 dimensiones
    graficar_espacio_latente(x_train, model)

    # muestro la comparacion de reconstruccion usando este modelo de cuello de botella angosto
    mostrar_comparacion(model, x_train)

    # devuelvo el modelo para poder usarlo despues, por ejemplo para generar caracteres nuevos
    return model


# grafico espacio latente
def graficar_espacio_latente(x_train, model):
    # armo un modelo nuevo que arranca igual que el original pero corta justo en la capa latente
    # asi puedo obtener directamente las 2 coordenadas de salida de esa capa, sin llegar hasta el final
    encoder = keras.Model(
        inputs=model.layers[0].input,
        outputs=model.get_layer("espacio_latente").output
    )

    # le paso todos los caracteres y obtengo sus coordenadas (x,y) en el espacio latente
    # me queda un array de forma (32, 2): 32 caracteres, 2 coordenadas cada uno
    coordenadas = encoder.predict(x_train)

    # armo el grafico de dispersion con esas coordenadas
    plt.figure(figsize=(10, 8))
    plt.scatter(coordenadas[:, 0], coordenadas[:, 1])

    # le pongo al lado de cada punto la letra que representa, para poder identificar visualmente cada uno
    for i, letra in enumerate(etiquetas):
        plt.annotate(letra, (coordenadas[i, 0], coordenadas[i, 1]))

    plt.title(f"Espacio latente 2D - {font}")
    plt.xlabel("Neurona latente 1")
    plt.ylabel("Neurona latente 2")
    plt.grid(True)
    plt.show()


# genera un carácter nuevo a partir de coordenadas inventadas
def generar_caracter(model, x, y):
    # creo una entrada nueva que va a recibir directamente las 2 coordenadas del espacio latente
    # no la imagen completa, sino el punto (x,y) que yo elija a mano
    input_latente = keras.Input(shape=(2,))

    # uso esta bandera para saber cuando ya pase la capa latente mientras recorro el modelo
    encontrado = False
    salida = input_latente

    # recorro todas las capas del modelo original en orden
    for capa in model.layers:
        # solo conecto las capas que vienen DESPUES de la capa latente
        # porque quiero armar unicamente la mitad decodificadora, sin la parte que comprime
        if encontrado:
            salida = capa(salida)
        # cuando encuentro la capa latente, prendo la bandera para que las siguientes capas si se conecten
        if capa.name == "espacio_latente":
            encontrado = True

    # armo el modelo decodificador que arranca en las 2 coordenadas y termina en la imagen reconstruida
    decodificador = keras.Model(inputs=input_latente, outputs=salida)

    # armo el punto que quiero probar, con la forma que espera keras: una lista dentro de una lista
    punto_nuevo = np.array([[x, y]])

    # le pido al decodificador que me genere la imagen correspondiente a ese punto inventado
    caracter_generado = decodificador.predict(punto_nuevo)
    matriz = np.squeeze(caracter_generado)

    # muestro el resultado dibujado en pantalla
    print(f"\n--- Caracter generado en coordenadas ({x}, {y}) ---")
    print("-----------------------------")
    for fila in range(7):
        dibujo = "".join(["█" if p > 0.5 else " " for p in matriz[fila]])
        print(f"  Fila {fila}:       {dibujo}")


def eliminar_ruido(x_train, model, prob_ruido=0.05, indice_caracter=1):

    # tomo el caracter original que quiero poner a prueba
    caracter_original = x_train[indice_caracter]

    # hago una copia para no modificar el caracter original
    x_con_ruido = caracter_original.copy()

    # genero una mascara booleana del mismo tamaño que el caracter
    # cada pixel tiene probabilidad prob_ruido de ser alterado
    mascara = np.random.rand(*x_con_ruido.shape) < prob_ruido

    # invierto todos los pixeles donde la mascara resulto verdadera
    # si era 1 pasa a 0 y si era 0 pasa a 1
    x_con_ruido[mascara] = 1 - x_con_ruido[mascara]

    # agrego una dimension extra para que keras reciba un lote de un unico caracter
    x_con_ruido = np.expand_dims(x_con_ruido, axis=0)

    # le paso el caracter con ruido al modelo entrenado
    # obtengo la reconstruccion realizada por el autoencoder
    prediccion = model.predict(x_con_ruido, verbose=0)

    print(f"\nELIMINACION DE RUIDO | prob_ruido={prob_ruido}")
    print(" ORIGINAL | CON RUIDO | RECONSTRUIDA")
    print("------------------------------------")

    # guardo por separado el caracter original, el caracter con ruido
    # y la reconstruccion realizada por la red
    matriz_original = caracter_original
    matriz_ruido = x_con_ruido[0]
    matriz_reconstruida = prediccion[0]

    # recorro las siete filas del caracter para imprimirlas una debajo de otra
    for fila in range(7):

        # convierto los valores numericos en un dibujo
        #
        # si el valor supera ese umbral considero que el pixel esta encendido
        orig = "".join("█" if p == 1 else " " for p in matriz_original[fila])
        ruido = "".join("█" if p == 1 else " " for p in matriz_ruido[fila])
        recon = "".join("█" if p > 0.5 else " " for p in matriz_reconstruida[fila])

        # muestro lado a lado el caracter original, la version con ruido
        # y la reconstruccion realizada por el autoencoder
        print(f"{orig:<8} | {ruido:<8} | {recon}")

# PUNTO 1

#obtengo los datos de entranamiento
x_train = cargar_datos(font)

# pruebo distintas arquitecturas variando capas y parametros pruebo una arquitectura simetrica con cuello de botella de 4 neuronas
#entrenar_evaluacion(x_train,[24, 12, 4, 12, 24], lr=0.005, epochs=300, batch=4)

# pruebo una arquitectura mas profunda, con mas capas intermedias y cuello de botella mas angosto
#entrenar_evaluacion(x_train,[28, 16, 8, 4, 8, 16, 28], lr=0.005, epochs=300, batch=4)

# pruebo bajar el learning rate y aumentar las epocas, comprimiendo hasta solo 2 neuronas
#entrenar_evaluacion(x_train,[20, 10, 2, 10, 20], lr=0.0005, epochs=600, batch=2)

# pruebo una arquitectura mas chica, con menos capas en total
#modelo_4 = entrenar_evaluacion(x_train, [20, 10, 20], lr=0.005, epochs=800, batch=4)


# entreno especificamente el modelo con latente de 2 dimensiones para poder graficarlo despues
#modelo_latente = entrenar_espacio_latente(x_train, lr=0.001, epochs=1500, batch=2)
# genero un caracter nuevo a partir de un punto que invento dentro del espacio latente
# elijo coordenadas que esten en una zona intermedia del grafico, entre dos letras conocidas
#generar_caracter(modelo_latente, x=-20, y=30)


#PUNTO 2
# entreno modelo donde de extra le paso el ruido para entrenamiento
modelo_5 = entrenar_evaluacion(x_train[:6], [20 ,10, 20], lr=0.0005, epochs=300, batch=16, prob_ruido=0.1)
# pruebo la capacidad del modelo de eliminar ruido,
eliminar_ruido(x_train[:6],modelo_5,0.1,indice_caracter=1) # 20% de ,modelo_5,0.02,indice_caracter=1) # 2% de píxeles invertidos
eliminar_ruido(x_train[:6],modelo_5,0.05,indice_caracter=2) # 5% de píxeles invertidos
eliminar_ruido(x_train[:6],modelo_5,0.08,indice_caracter=3) # 8% de píxeles invertidos
eliminar_ruido(x_train[:6],modelo_5,0.1,indice_caracter=4) # 10% de píxeles invertidos
eliminar_ruido(x_train[:6],modelo_5,0.2,indice_caracter=5) # 20% de píxeles invertidos
