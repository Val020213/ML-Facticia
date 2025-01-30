# Colección Facticia de Emilio Roig de Leuchsenring

## Problema

Este proyecto surge como colaboración de la Universidad de la Habana con el departamento de digitalización de
edificio Santo Domingo, comúnmente conocido como San Geronimo por la Universidad que allí se encuentra.
El objetivo de este proyecto es extraer las imágenes de los documentos de la colección facticia, legada
por Emilio Roig de Leuchsenring, y almacenarlas en un formato digital junto con su descripción, para agilizar
el proceso de digitalización de los documentos.

## Estado del Arte

La gran parte de las investigaciones en el campo de la extracción de imágenes y texto se han centrado en el uso de modelos de
aprendizaje profundo (`deep learning`) para la detección de objetos en imágenes y la extracción de texto en documentos. Sin embargo
ambas tareas han sido complicada por la heterogeneidad de imágenes, tipografías de los textos, orientación de los textos, flujo y maquetación
de los documentos.

En el caso general de la segmentación de imágenes, se han utilizado modelos de aprendizaje profundo como YOLO v8, por sus grandes resultados
obtenidos, su gran documentación y soporte por la comunidad de código abierto (`open source`). Entre los problemas que se han resuelto con YOLO
se encuentran la detección de matrículas vehiculares, el análisis de publicidad (Wang et al.,2024)[1]. YOLO además posee gran variedad de arquitecturas
según el problema a resolver, y con diferentes niveles de parámetros para ajustar la precisión y velocidad del modelo. En paralelo la extracción de texto
se ha mejorado con el uso de motores OCR (`Optical Character Recognition`, en español, Reconocimiento óptico de caracteres) como EasyOCR, EfficientOCR y Calamari, que permiten transformar información visual en datos legibles y estructurados (Skelbye & Dannélls, 2021)[2] y proporcionando benchmarks confiables para medir su desempeño (Du et al., 2024) [4].

Para la asociación de imágenes y texto, se han propuesto modelos como MKL-VisITA
que integran Multi-Kernel Learning y Vision Transformers para mejorar la representación compartida de imágenes y texto, teniendo un
buen rendimiento con los conjuntos de datos MSCOCO (Microsoft Common Objects in Context) y Flickr30K (dataset de imágenes donde cada imagen tiene cinco frases de referencias etiquetadas por los humanos)(Wang et al.,2024)[3].

Un aspecto fundamental dentro de estos procesos es el preprocesamiento de datos, que influye directamente en la
calidad de los resultados obtenidos. En el caso de las imágenes, se han implementado técnicas como la binarización,
la eliminación de ruido y la normalización del color. Para el texto, se han aplicado estrategias de tokenización, vectorización y padding,
lo que facilita su procesamiento por parte de los modelos de aprendizaje profundo (Skelbye & Dannélls, 2021) [2].
**_ AQUÍ PROFUNDIZAR EN QUE TÉCNICAS O NOMBRES ESPECÍFICOS _**

Centrándonos en las investigaciones realizadas, podemos determinar que existen diferentes enfoques para resolver el problema de reconocimiento y extracción de textos e imágenes, y asociar los datos extraídos. Sin embargo, no existe un modelo único que resuelva todo el problema, por lo que la solución debe ser una combinación de varios modelos, cada uno especializado en una tarea. Otro punto clave es la calidad de los datos, ya que la mayoría de las investigaciones se han realizado con datos de buena calidad, no están públicos para su uso, siendo escasas las investigaciones con datos de colecciones antiguas y deterioradas.

**_ INSERTAR TABULACIÓN DEL ESTADO DEL ARTE _**

** HABLAR SOBRE CLIP Y TESSARACT **
** HABLAR DE OTROS MODELOS DE RECONOCIMIENTO DE IMÁGENES **

## Conjunto de Datos

El conjunto de datos consiste en 220 carpetas que contienen entre 30 - 400 fotografías cada una. Estas fotografías están en un orden especifico, preservando la estructura de los documentos originales, por lo cual es útil preservar este dato en la extracción de datos sobre ellas, para encontrar otros metadatos de interés de los historiadores. Las fotografías están en `jpg` con una resolución mayor a `1080px` en ambas dimensiones.

Pueden contener textos horizontales, verticales, o una combinación de ambos en una misma imagen. De igual forma una imagen puede contener textos en diferente fuente, tamaño, interlineado y grosor.

**\* IMAGENES DE EJEMPLO**

El color del papel o color de fondo de las fotografías puede variar entre blanco a tonalidades de amarillo, incluso rojizas en algunos casos. Los textos en su mayoría son mecanografiados, aunque existen tachaduras y escritos a manos en algunos casos, como suelen ser fechas, firmas y anotaciones marginales, número interno de la página, entre otros. El texto está en español, aunque en ligeros casos puede contener palabras en otros idiomas, pero del alfabeto latino.

**\* IMAGNES DE EJEMPLO**

Las fotografías pueden ser imágenes de una página completa, fragmentos de una página, pueden estar en una columna de texto, u horizontales a lo largo de la página. Las imágenes pueden ser dibujos, gráficos, mapas, o fotografías de personas, pueden estar rodeadas de algún contorno o marco que las delimite. En su mayoría las imágenes son con forma rectangular, aunque existen casos de imágenes con formas irregulares.

**\* IMAGENES DE EJEMPLO**

Debido a que estas fotografías son de documentos antiguos, pueden contener manchas, dobleces, arrugas, desgaste del papel, y otros daños que afectan la calidad de la imagen. En algunos casos las fotografías pueden estar rotadas en un ángulo, pero siempre con una perspectiva frontal. Y es requisito del problema recuperar las imágenes en el ángulo correcto, para evitar que el investigador tenga que rotar la imagen o recortarla para la digitalización.

**\* IMAGENES DE EJEMPLO **

El conjunto de datos pesa alrededor de 60GB en total, es de acceso publico y puede ser descargado desde las Paginas oficiales de la oficina del historiador de la Habana **\*INSERTAR ENLACE**.

## Propuesta de solución

El problema de correspondencia entre imágenes y texto en documentos históricos presenta desafíos en la localización de contenido dentro de las fotografías y su posterior asociación con la información correcta. Para abordar esta problemática, hemos dividido nuestro enfoque en dos subproblemas principales: (1) Detección y extracción de texto y (2) Asociación de imágenes con texto.

### Detención de Imágenes y Textos en las fotografías

Para identificar imágenes y textos en las fotografías decidimos utilizar el modelo de Deep Learning YOLOv11, en su version mas reciente(YOLOv11). YOLO posee la variante OBB(oriented bounding boxes - cuadros delimitadores orientados) que permite cumplir con el requisito de identificar las imágenes según el angulo que ha sido tomada la fotografía, además rotar el texto a su alineación original facilita la integración con los OCR. Permite identificar varias clases diferentes en un mismo análisis, lo que nos facilita no tener dos modelos para identificar texto e imágenes por separado. EL modelo puede refinarse en el futuro añadiendo nuevas fotografías para mejorar las métricas en los escenarios que sean de interés, de igual manera nos permite usar modelos pre entrenados y mejorarlos con nuestros datos. YOLO al ser un modelo de deep learning tiene la capacidad de reconocer patrones complejos en las fotografías, como textos deformados, fondos borrosos o imágenes movidas. YOLO utiliza técnicas de multi-escala (ej: FPN - Feature Pyramid Networks) para detectar objetos pequeños, lo cual puede ser util para columnas estrechas y pie de fotos. Además YOLO tiene varios modelos que permiten ajustar la precisión y velocidad del modelo, en nuestro caso usamos el modelo Nano que permite procesar grandes volúmenes de imágenes en un tiempo relativamente rápido sin tener que usar gpu costosas, lo cual es ideal para el desarrollo de un software con los pesos exportados.

Para el entrenamiento se etiquetaron en total 1221 imágenes, etiquetando 4 clases principales:

- Texto: Todo tipo de textos mecanografiados, encabezados siempre manteniendo su dirección original según el flujo de la página.
- Imagen: Todo tipo de imágenes, dibujos, gráficos, mapas, fotografías, etc.
- Escrito a mano: Anotaciones marginales, firmas, fechas, números internos de la página, etc.
- Caption: Textos que acompañan a las imágenes, como títulos, subtítulos, pies de foto. Esta clase lo hicimos como extra para poder asociar images con sus descripciones. Los captions usualmente son textos cortos y están cerca de las imágenes, en negrita o en cursiva. Para mejorar la selección de captions, decidimos llevar el bounding box desde el texto con un filo pequeño de la imagen.

De las 1221 fotografías etiquetadas obtuvimos: 3492 textos, 794 imágenes, 981 captions y 483 escritos a mano.

### Experimentos en la detección de imágenes y textos en fotografías

Para optimizar los resultados de la detección de imágenes y textos en las fotografías, nuestro primer paso fue ajustar los hiperparametros de YOLO (hyperparameter tuning), se uso una configuración de 30 iteraciones probando distintas configuraciones. El procesamiento duro alrededor de 12 horas usando una maquina virtual de Kaggle, usando dos GPU T4, y 32 GB de RAM. Manteniendo el procesamiento a 1080px que es lo máximo que YOLO puede procesar. 

Los resultados fueron los siguientes:

** insertar hypeparametros **

Las métricas al entrenar el modelo con esos hiperparametros fueron:

** INSERTAR RESULTADOS Y EXPLICAR LAS METRICAS POR ARRIBA **


** AQUI HABLAR DE LOS RESULTADOS Y ANALISIS CRITICO DE LOS RESULTADOS **

### Otros experimentos

preprocesamiento de las imagenes con escalas de grises y binarización, para mejorar la detección de texto en las imagenes.
hyperparametrizacion y muestra de los resultados del entrenamiento

### Asociación de imágenes y texto

*** Preprocesamiento ***

*** TESSERACT ***

*** AQUI HABLAR DE LAS METRICAS *** CER Y JACKARD

*** post procesamiento ***

*** CLIP ***

*** AQUI HABLAR DE LOS RESULTADOS Y ANALISIS CRITICO DE LOS RESULTADOS ***


