## 🧪 Resultados de los Experimentos

A continuación se registran los resultados obtenidos al ejecutar el entrenamiento variando el número de épocas y la tasa de aprendizaje (usando la semilla `seed=7`):

### 1. Prueba base (10,000 épocas | Tasa de aprendizaje: 0.5)
* **El error final (MSE):** `0.019645`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5490`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Bien y estable**. Aprendió sin problemas y clasificó todo bien.

---

### 2. Pocas épocas (100 épocas | Tasa de aprendizaje: 0.5)
* **El error final (MSE):** `0.165623`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5217`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Muy poquito tiempo**. Acertó a las respuestas pero el error quedó alto y las probabilidades quedaron con dudas.

---

### 3. Cantidad intermedia (1,000 épocas | Tasa de aprendizaje: 0.5)
* **El error final (MSE):** `0.067481`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5793`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Aceptable**. Le atinó a todo pero todavía podía mejorar un poco más.

---

### 4. Más épocas (20,000 épocas | Tasa de aprendizaje: 0.5)
* **El error final (MSE):** `0.011705`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5263`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Muy preciso**. Al darle más vueltas bajó el error casi a la mitad con respecto a la prueba base.

---

### 5. Tasa pequeña (10,000 épocas | Tasa de aprendizaje: 0.01)
* **El error final (MSE):** `0.130048`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5388`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Muy lento**. Daba pasos muy chiquitos y no le alcanzó el tiempo para bajar más el error.

---

### 6. Tasa moderada (10,000 épocas | Tasa de aprendizaje: 0.1)
* **El error final (MSE):** `0.049728`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5853`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Algo lento pero seguro**. Aprendió despacio pero logró clasificar todo bien.

---

### 7. Tasa alta (10,000 épocas | Tasa de aprendizaje: 1.0)
* **El error final (MSE):** `0.011705`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.5082`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Rápido**. Aprendió rápido y bajó el error sin enredarse.

---

### 8. Tasa muy alta (10,000 épocas | Tasa de aprendizaje: 2.0)
* **El error final (MSE):** `0.006566`
* **La cantidad de respuestas correctas en los diez datos de entrenamiento:** `10 / 10` (100%)
* **La probabilidad obtenida para humedad 45% y temperatura 34 °C:** `0.8540`
* **Si el aprendizaje fue rápido, lento, inestable o insuficiente:** **Súper rápido**. Avanzó a pasos gigantes y logró resultados muy seguros rápidamente.

---

## 📝 Análisis del Problema y Preguntas

### 1. ¿Por qué fue necesario normalizar la humedad y la temperatura?
Porque ambas entradas manejaban escalas numéricas sustancialmente diferentes (Humedad entre 0 y 100, Temperatura entre 0 y 50). Si no se normalizan, la variable con valores más altos domina el cálculo del gradiente, haciendo que la neurona aprenda de forma ineficiente o inestable.

### 2. ¿En qué operaciones se utilizó `X_normalizado` y para qué se conservó `X`?
* `X_normalizado` se usó para realizar todos los cálculos del entrenamiento: para multiplicar las entradas por los pesos (`z = X_norm @ pesos + sesgo`) y para calcular los cambios en los pesos (`X_norm.T @ gradiente_z`).
* `X` se guardó con sus valores reales (porcentajes de humedad y grados Celsius) únicamente para mostrar los datos de forma clara y legible en la consola.

### 3. ¿Qué ocurrió al utilizar solamente 100 épocas?
Aunque la neurona logró clasificar correctamente los 10 datos de entrenamiento (10/10) gracias al umbral de 0.5, las probabilidades calculadas por la función sigmoide quedaron en un punto muy cercano a la duda (alrededor de 0.5217) y el error cuadrático medio permaneció elevado (`0.165623`). Esto demuestra que 100 épocas fueron insuficientes para consolidar un aprendizaje preciso.

### 4. ¿Más épocas siempre produjeron una mejora importante?
No necesariamente. Aumentar de 10,000 a 20,000 épocas continuó reduciendo el MSE (de `0.019645` a `0.011705`), pero con rendimientos decrecientes (el porcentaje de aciertos no cambió porque desde las 100 épocas ya acertaba 10/10). Esto sucede porque la función sigmoide se satura cerca de 0 y 1, haciendo que los gradientes se vuelvan cada vez más pequeños.

### 5. ¿Qué efecto tuvo una tasa de aprendizaje demasiado pequeña (0.01)?
Provocó que las actualizaciones de los pesos en cada época fueran diminutas. Aunque con 10,000 épocas alcanzó a clasificar los 10 datos correctamente (10/10), el error cuadrático medio quedó bastante alto (`0.130048`), mostrando que el aprendizaje fue muy lento y le faltaron épocas para optimizarse más.

### 6. ¿Qué efecto tuvo una tasa de aprendizaje alta o muy alta (1.0 o 2.0)?
Aceleró drásticamente el proceso de aprendizaje y redujo el error a valores mínimos en menos tiempo (con tasa 2.0 el MSE bajó hasta `0.006566`). En un modelo simple como este no generó problemas, aunque en redes más complejas tasas muy altas pueden hacer que el modelo oscile o no converja.

### 7. ¿Qué representa el signo del peso correspondiente a la humedad?
Un peso con signo **negativo** (aprox. `-19.5`). Representa una **relación inversamente proporcional**: a mayor humedad disponible en el suelo, menor es la necesidad o probabilidad de activar el riego.

### 8. ¿Qué representa el signo del peso correspondiente a la temperatura?
Un peso con signo **positivo** (aprox. `+2.3`). Representa una **relación directamente proporcional**: a mayor temperatura ambiental, mayor es la evaporación y, por ende, mayor la necesidad o probabilidad de regar.

### 9. ¿Por qué una probabilidad debe convertirse en 0 o 1 mediante un umbral?
Porque la activación de un sistema físico de riego (como una electroválvula o bomba de agua) requiere un **comando discreto/binario**: Encendido (`1`) o Apagado (`0`). La función sigmoide produce una probabilidad continua entre 0 y 1, por lo que se utiliza un umbral (usualmente `0.5`) para tomar la decisión final.

### 10. ¿Qué limitaciones tiene esta neurona para representar el riego de una planta real?
1. **Modelado lineal:** Un solo perceptrón solo puede aprender límites de decisión linealmente separables.
2. **Factores omitidos:** En la agricultura real influyen muchas más variables como el tipo de cultivo, la humedad relativa del aire, la radiación solar, la fase de crecimiento de la planta y la hora del día.
3. **Datos sintéticos:** Los datos de entrenamiento son didácticos y no provienen de una calibración agronómica real.
