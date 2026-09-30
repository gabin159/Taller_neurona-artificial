import numpy as np

print(
    "Neurona sencilla para que aprenda el comportamiento del riego de plantas"
)

# Datos de entrenamiento originales: [Humedad (%), Temperatura (°C)]
X = np.array(
    [
        [80, 18],
        [70, 22],
        [65, 28],
        [55, 25],
        [50, 32],
        [40, 30],
        [35, 25],
        [30, 32],
        [20, 35],
        [10, 38],
    ],
    dtype=float,
)  # Usamos float para permitir decimales en los cálculos[cite: 1]

# Respuestas esperadas (Salida objetivo): 0 = No regar, 1 = Regar
y = np.array(
    [[0], [0], [0], [0], [0], [1], [1], [1], [1], [1]], dtype=float
)  # Matriz columna con la etiqueta deseada[cite: 1]

# Normalización: ajusta los datos a una escala homogénea entre 0 y 1[cite: 1]
escala = np.array(
    [100, 50]
)  # Humedad máxima estimada ~100%, Temp ~50°C[cite: 1]
X_normalizado = (
    X / escala
)  # Dividimos cada columna por su escala para estabilizar el aprendizaje[cite: 1]


# Función de activación: mapea cualquier valor z a un rango de probabilidad entre 0 y 1[cite: 1]
def sigmoide(z):
    return 1 / (1 + np.exp(-z))


# Función para realizar el proceso de entrenamiento de la neurona[cite: 1]
def entrenar_neurona(
    X_norm, y, tasa_aprendizaje=0.5, epocas=10000, seed=7, verbose=False
):
    # Generador de números aleatorios para inicializar los pesos[cite: 1]
    rng = np.random.default_rng(seed)
    pesos = rng.normal(
        size=(2, 1)
    )  # Se asigna un peso inicial aleatorio para la humedad y otro para la temperatura[cite: 1]
    sesgo = 0.0  # El sesgo (bias) inicializa en cero[cite: 1]

    # --- LÍNEA PARA MOSTRAR LA CONFIGURACIÓN DE ÉPOCAS Y TASA DE APRENDIZAJE ---
    if verbose:
        print(
            f"Configuración de entrenamiento: Épocas = {epocas} | Tasa de aprendizaje = {tasa_aprendizaje}"
        )
        print("-" * 65)

    # Determinamos el intervalo de impresión según el total de épocas
    intervalo = max(1, epocas // 5)

    # Bucle de entrenamiento que se repite 'epocas' veces[cite: 1]
    for epoca in range(epocas):
        # 1. Propagación hacia adelante (Forward Propagation):
        # Suma ponderada: z = (humedad_norm * w1) + (temp_norm * w2) + sesgo
        z = X_norm @ pesos + sesgo  # '@' realiza la multiplicación matricial
        predicciones = sigmoide(
            z
        )  # Aplicamos la activación para obtener probabilidades de 0 a 1[cite: 1]

        # 2. Cálculo del Error (MSE derivative) y Retropropagación (Backpropagation):
        error = predicciones - y  # Diferencia entre lo predecido y lo real[cite: 1]

        # Derivada de la regla de la cadena para la función sigmoide y la pérdida MSE[cite: 1]
        gradiente_z = (
            2 * error * predicciones * (1 - predicciones) / len(X_norm)
        )  # Gradiente con respecto a z[cite: 1]
        gradiente_pesos = (
            X_norm.T @ gradiente_z
        )  # Producto punto para saber cuánto debe cambiar cada peso[cite: 1]
        gradiente_sesgo = np.sum(
            gradiente_z
        )  # Suma de los gradientes para ajustar el sesgo[cite: 1]

        # 3. Actualización de Parámetros (Descenso del Gradiente):
        # Ajustamos pesos y sesgo en dirección opuesta al gradiente[cite: 1]
        pesos -= tasa_aprendizaje * gradiente_pesos
        sesgo -= tasa_aprendizaje * gradiente_sesgo

        # Muestra el avance del error dinámicamente según la cantidad de épocas
        if verbose and (epoca % intervalo == 0 or epoca == epocas - 1):
            perdida = np.mean(error**2)
            print(f"Época {epoca:5d} | Error MSE: {perdida:.6f}")

    # Calcula el error final al terminar todas las épocas
    error_final = np.mean((sigmoide(X_norm @ pesos + sesgo) - y) ** 2)
    return pesos, sesgo, error_final


if __name__ == "__main__":
    print("\n--- ENTRENAMIENTO DE LA NEURONA ---")
    pesos, sesgo, error_final = entrenar_neurona(
        X_normalizado, y, tasa_aprendizaje=0.5, epocas=10000, verbose=True
    )

    print("\nResultados del modelo entrenado:")
    print(f"Peso Humedad: {pesos[0][0]:.4f}")
    print(f"Peso Temperatura: {pesos[1][0]:.4f}")
    print(f"Sesgo (Bias): {sesgo:.4f}")
    print(f"Error Final (MSE): {error_final:.6f}")

    # --- RESPUESTAS CORRECTAS EN DATOS DE ENTRENAMIENTO ---
    probs_entrenamiento = sigmoide(X_normalizado @ pesos + sesgo)
    preds_entrenamiento = (probs_entrenamiento >= 0.5).astype(int)
    correctas = np.sum(preds_entrenamiento == y.astype(int))
    print(
        f"Respuestas correctas en entrenamiento: {correctas} / {len(y)} ({correctas/len(y)*100:.0f}%)"
    )

    # Pruebas con condiciones totalmente nuevas
    print("\n--- PREDICCIÓN CON NUEVOS CASOS ---")
    nuevos_datos = np.array(
        [[75, 30], [45, 34], [25, 22], [50, 25], [30, 40]], dtype=float
    )  # [Humedad, Temperatura][cite: 1]

    # ¡Importante! Los datos nuevos deben normalizarse con la MISMA escala utilizada en el entrenamiento[cite: 1]
    nuevos_norm = nuevos_datos / escala
    probs_nuevas = sigmoide(nuevos_norm @ pesos + sesgo)

    # Convertimos la probabilidad continua en una decisión binaria con un umbral de 0.5[cite: 1]
    decisiones = (probs_nuevas >= 0.5).astype(int)

    for orig, prob, dec in zip(nuevos_datos, probs_nuevas.ravel(), decisiones.ravel()):
        print(
            f"Humedad: {orig[0]:2.0f}% | Temp: {orig[1]:2.0f}°C -> "
            f"Probabilidad: {prob:.4f} | Decisión: {'REGAR (1)' if dec == 1 else 'NO REGAR (0)'}"
        )