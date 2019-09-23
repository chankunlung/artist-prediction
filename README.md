# Clasificación de los 50 mejores artistas de la historia

## Introducción
El problema presentado consiste en clasificar exitosamente obras de artes en 50 artistas. Para esto se utilizará una red neuronal artificial, particularmente se utilizara el modelo DenseNet. Se utilizó data augmentation en las imagenes para reducir overfitting y los efectos del desbalance en la cantidad de imágenes por clases. Se utilizará la librería tensorflow 2. 

## Análisis de los datos
Se utilizó el dataset Best Artworks of All Times descargado de kaggle, el cual posee 8446 imágenes de 50 de los mejores artistas de la historia. Estas imágenes poseen distinto tamaño entre sí y la distribución de imágenes en artistas no es uniforme, habiendo artistas con menos de 25 imagenes y artistas con más de 800.
Las imágenes fueron separadas en datos de entrenamiento y de testeo con una proporción de 80% y 20% respectivamente. Luego las imágenes fueron redistribuidas en subdirectorios para entrenamiento y testeo para poder ser recuperadas en el posterior entrenamiento.
Posteriormente se aumentaron la cantidad de imágenes en la carpeta de entrenamiento de aquellos artistas que poseyeran menos de 80 obras. Esto se realizó duplicando las obras y aplicandoles una ligera rotación, cambio de ancho y alto, volteo horizontal y vertical y zoom a las copias. De esta forma se reduce la diferencia en obras de aquellos artistas que poseen menos obras con aquellos que más poseen y se conserva la distribución de obras de cada artista al duplicar todas sus obras.
Las imágenes presentan diferentes estilos de arte. Incluso obras del mismo artista también pueden poseer diferentes estilos. Un ejemplo de este tipo de obras es Leonardo Da Vinci, quien posee tanto pinturas como dibujos en hojas.

![Da Vinci 1](https://github.com/okason97/artist-prediction/blob/master/images/Leonardo_da_Vinci_23.jpg)
![Da Vinci 2](https://github.com/okason97/artist-prediction/blob/master/images/Leonardo_da_Vinci_121.jpg)
               
## Data augmentation
Data augmentation consiste en aumentar artificialmente la cantidad de datos disponibles en base a los datos originales. Se introducen cambios sobre los datos originales para generar nuevos datos diferentes que serán utilizados en el entrenamiento. Data augmentation es muy útil para prevenir overfitting. 
Se tiene overfitting cuando la red neuronal se sobreajusta a los datos de entrenamiento lo cual la perjudica a la hora de clasificar nuevos datos no contenidos en los datos de entrenamiento. Una forma de medir el overfitting de una red neuronal es utilizando un conjunto de datos de test con los cuales la red no se entrenará, sino que se realizará inferencia sobre estos datos desconocidos para la red midiendo que tan bien esta los clasifica.
Debido a la gran cantidad de obras estas no pudieron ser cargadas al mismo tiempo en memoria, por lo que fue necesario procesar las imágenes en batches. Se utilizó ImageDataGenerator de keras para realizar el procesamiento del data augmentation y la separación del dataset en batches.
Se aplicó a las imágenes una rotación máxima de 20 grados, un estiramiento horizontal y vertical máximo del 20%, volteo horizontal y vertical y un zoom del 50%.
El tamaño de las imágenes se cambio para que todas poseen la misma forma de 512x512. De esta manera no se aumenta en sobremedida el requerimiento de memoria y tiempo de procesamiento pero se conserva una suficiente resolución para no perder los detalles de las obras.
Finalmente se utilizó Welford's online algorithm para el cálculo de la media y la varianza para aplicar normalización a las imágenes x restando la media y dividiendo por la varianza.

![z normalization](https://github.com/okason97/artist-prediction/blob/master/images/ecuation.png)

Ejemplo de una imagen obtenida utilizando data augmentation:

![normalized image](https://github.com/okason97/artist-prediction/blob/master/images/data_aug.png)

## Modelo
Como modelo fue seleccionado DenseNet, este modelo estado del arte ha demostrado en múltiples ocasiones ser uno de los mejores modelos a la hora de clasificar imágenes. Obteniendo un error rate de 17.18% en la clasificacion sobre el dataset Cifar100, un mejor error rate que otras redes state of the art como ResNet.
DenseNet es una red convolucional, esto significa que está compuesta por capas convolucionales. Una capa convolucional funciona mediante la utilización de filtros convolucionales, los cuales son un conjunto de matrices las cuales se irán deslizando por los features de entrada multiplicando sus valores para así obtener los datos que irán en el output de la capa. 

![conv](https://github.com/okason97/artist-prediction/blob/master/images/cnn.png)

Como es observable en la imagen de arriba el output de la capa convolucional tendrá un menor tamaño que el input a menos que se introduzca un padding. Esta reducción en la complejidad del input trae la ventaja de permitir reducir el costo computacional y de memoria de la red a medida que se avanza por esta.
Normalmente en las redes convolucionales el output de una capa se utilizará como el input de la siguiente capa.
DenseNet funciona mediante la utilizacion de dense blocks. Estos bloques están compuestos de bloques convolucionales cuyo input será el output de los bloques convolucionales anteriores del bloque dense concatenados. De esta manera el estado del bloque dense es completamente observable por este preservando el gradiente. 
Cada bloque convolucional está compuesto por una bottleneck layer, formada por una capa convolucional de 1x1 que ayuda a reducir la complejidad de la red, y una función compuesta. La función compuesta está formada por 3 operaciones: batch normalization, ReLU y una capa convolucional de 3x3.
DenseNet está compuesta por múltiples dense blocks con capas de transición entre estos. Estas capas de transición se encargan de realizar downsampling reduciendo el tamaño de los feature maps. Esta reducción de dimensionalidad es lograda utilizando una capa bottleneck seguida por una capa de average pooling de 2x2. 
Para volver más compacto el modelo se utiliza un factor de compresión en las capas de transición. De esta forma si el input de una capa de transición es un feature map de tamaño m y usamos un factor de compresión 0<t<1, el output de la capa de transición sera un feature map de tamaño tm.

![dense](https://github.com/okason97/artist-prediction/blob/master/images/dense.png)

Utilizando este modelo, la red neuronal puede ser menor en tamaño que otros modelos state of the art pero brindando un mejor resultado que otros modelos state of the art. Ser más compacta otorga múltiples beneficios. Una red más compacta ocupa menos memoria pudiendo ser entrenada sin la necesidad de utilizar hardware demasiado caro o servicios en la nube. A su vez este tipo de red permiten mitigar el efecto del vanishing gradiant el cual surge al poseer una red muy profunda donde el entrenamiento afectará en menor medida a las capas del inicio. Esta posible reducción de tamaño es gracias a la reutilización de features a lo largo de la red introducida por la concatenación de inputs.
A este modelo se le agregan capas de Squeeze and Excitation (SE). Estas capas se concentran en la informacion channel wise obtenida de las capas convolucionales. SE realiza recalibracion en las features mejorando la calidad de las representaciones obtenidas de la red modelando la interdependencia entre canales.
Un bloque SE funciona como una operación de 2 pasos: squeeze y excitation. 
El paso squeeze consta de una capa average pooling, con la cual se reducirá la dimensionalidad del feature de entrada a un vector de 1 dimensión de tamaño V, siendo V la cantidad de canales que poseía el feature de entrada.
Luego se realizará el paso de excitation, el cual capturará las dependencias entre canales en una manera flexible y no mutuamente exclusiva. Esto lo realiza utilizando 2 capas densely connected donde la primera tendrá una función de activación ReLU y la segunda una función de activación sigmoide para evitar la exclusión de canales. El resultado de las capas densely connected será un vector con la escala que se le aplicará a cada canal. Como paso final se multiplicará este vector de escala al feature map de entrada original obteniendo finalmente un output cuyos canales serán pesados según la interdependencia entre estos.
Se aplicaron SE blocks luego de capa bloque convolucional y de transición.

![sedense](https://github.com/okason97/artist-prediction/blob/master/images/sedense.png)

## Función de costo
Se utilizó cross-entropy como función de costo, a medida que el valor original se aleja del valor predicho, se obtendrá mayor pérdida.
Debido al desbalance en la cantidad de imágenes por clase, fue introducido una pérdida pesada. Esto se hizo asignando un peso a cada clase del conjunto de datos según la cantidad de imágenes que este disponga en el entrenamiento. Luego de obtener este peso, se le aplicará a la función de pérdida según la clase correcta a predecir, de forma tal que aquellas clases que posean menos datos afecten en mayor medida al modelo que aquellas que poseen mayor cantidad de datos.

## Optimizador
Se utilizó un optimizador Adam con learning rate inicial de 0.01. Este algoritmo es una variación del algoritmo de reducción de gradiente estocástico.
A diferencia del stochastic gradient descent clásico que utiliza un único learning rate fijo a través de todo el entrenamiento de la red, el algoritmo Adam posee un learning rate por parámetro el cual se actualizará a medida que el entrenamiento avanza según cuan rapido cambian los pesos.

## Entrenamiento
Se entrenó el modelo usando paciencia de 40. La paciencia indicará cuántos epochs el modelo hará sin reducir la pérdida hasta detenerse. 
Se utilizó un tamaño de batch de 8, ya que mayor tamaño de batch no cabía en la memoria de la GPU.
Las imágenes fueron tomadas directamente de los directorios de a batches y luego se les aplicó data augmentation y normalización debido a que estas no cabía en memoria.
Se utilizó tensorboard para monitorear los cambios en el accuracy y la pérdida del modelo en el entrenamiento y para almacenar una matriz de confusión para los datos de testeo.
El entrenamiento se realizó sobre una GPU NVidia RTX 2060 con una CPU Intel core I7 8700 y 8 gb de ram. Este entrenamiento llevo un tiempo aproximado de 17 horas.
Se probaron varios tamaños del modelo hasta llegar al que otorgó mejores resultados siendo este DenseNet con un growth rate de 16 (tasa de crecimiento aplicada a los filtros a medida que se avanza por la red), una cantidad de capas de [6, 12, 24, 16] y una reducción de 0.5 en sus bloques de transición.
Resultados
El mayor accuracy en test fue alcanzado en el step 106 (modelo A) con un valor de 62.96.
El modelo con menor loss en test fue alcanzado en el step 68 (modelo B) con accuracy en test de 61.48.
Matriz de correlación del modelo A:

![matrixa](https://github.com/okason97/artist-prediction/blob/master/images/modeloa.png)

Matriz de correlación del modelo B:

![matrixb](https://github.com/okason97/artist-prediction/blob/master/images/modelb.png)

A continuación se presentan gráficos para mostrar la reducción en la pérdida y el aumento de accuracy en entrenamiento y test:

![acc](https://github.com/okason97/artist-prediction/blob/master/images/acc.png)
![loss](https://github.com/okason97/artist-prediction/blob/master/images/loss.png)

## Conclusiones
El problema presentado era complejo debido a la variabilidad en el tipo de imágenes dentro de las imágenes de algunos artistas y de la similaridad en las imágenes de obras de diferentes artistas, y a la gran diferencia entre cantidad de obras de los diferentes artistas. 
Esta gran diferencia en cantidad de obras entre artistas es un problema difícil de afrontar debido a que cada artista hizo una cantidad de obras limitadas, por lo que es imposible añadir nuevas obras de un artista al dataset intentando equilibrar una vez que se ha añadido cada obra de este artista. Para un trabajo futuro se podría intentar implementar una GAN para buscar generar nuevas obras para aquellos artistas que posean menos obras.
En el modelo A se puede observar como el modelo presenta dificultades distinguiendo entre Eugene Delacroix y Peter Paul Rubens, prediciendo Peter Paul Rubens en lugar de Eugene Delacroix en múltiples ocasiones.

Obra de Eugene Delacroix

![Delacroix](https://github.com/okason97/artist-prediction/blob/master/images/delacroix.png)

Obra de Peter Paul Rubens

![Rubens](https://github.com/okason97/artist-prediction/blob/master/images/rubens.png)

Se puede observar que ambos artistas presentan estilos de arte y colores en sus obras similares. Esto junto al hecho de que Eugene Delacroix posee menos obras contribuye a la mala clasificación.
El modelo B confunde principalmente a las obras de Edvard Munch con las obras de Amedeo Modigliani, prediciendo Amedeo Modigliani en ocasiones donde la obra es de Edvard Munch. En esta ocasión Amedeo Modigliani es quien posee más obras entre ambos.

Obra de Edvard Munch

![Munch](https://github.com/okason97/artist-prediction/blob/master/images/munch.png)

Obra de Amedeo Modigliani

![Modigliani](https://github.com/okason97/artist-prediction/blob/master/images/modigliani.png)

Se puede observar que el estilo del arte es similar, esto sumado a la mayor cantidad de ejemplos de Amedeo Modigliani es lo que genera la confusión.
A pesar de la dificultad del dataset se alcanzó una buena precisión para el problema dado con un máximo de 62.96 de accuracy.
