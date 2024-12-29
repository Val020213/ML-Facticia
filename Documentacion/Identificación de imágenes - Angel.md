# Propuesta 3

## Identificación de imágenes

- **[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://paperswithcode.com/paper/yolov4-optimal-speed-and-accuracy-of-object)**
    - **YOLOv4: Optimal Speed and Accuracy of Object Detection**
    - **Resumen:**
        
        El artículo presenta **YOLOv4 (You Only Look Once v4)**, un modelo de aprendizaje profundo que combina precisión y velocidad para la detección de objetos en imágenes en tiempo real. El objetivo principal es optimizar tanto el rendimiento como la eficiencia, haciéndolo adecuado para aplicaciones prácticas como vigilancia, conducción autónoma, y más.
        
        ### **Modelos utilizados y herramientas adicionales**
        
        - **Modelo principal: YOLOv4**
            - Arquitectura optimizada basada en convoluciones profundas para realizar detecciones rápidas y precisas.
        - **Componentes innovadores:**
            - **CSPDarknet53:** Backbone del modelo para extracción de características.
            - **Spatial Pyramid Pooling (SPP):** Mejora la detección de objetos de diferentes tamaños.
            - **Path Aggregation Network (PAN):** Ayuda a combinar características de diferentes niveles.
        
        ### **Técnicas de extracción de características**
        
        - **Anchor Boxes:** Se utilizan para predecir las coordenadas de los objetos.
        - **Data augmentation:** Técnicas como mosaico, aumentos en perspectiva y cambios de escala para mejorar la robustez del modelo.
        - **Bag of Freebies (BoF):** Métodos para mejorar el rendimiento sin aumentar el costo computacional (e.g., regularización, ajustes en los anclajes).
        
        ### **Dataset**
        
        - YOLOv4 se probó en **COCO dataset**, un estándar en el campo que incluye 80 clases de objetos y millones de imágenes anotadas.
        
        ### **Resultados y métricas**
        
        - **Mean Average Precision (mAP):** Alcanzó un 43.5% en el conjunto de datos COCO, superando a otros modelos similares.
        - **Velocidad:** Procesa 65 FPS en GPUs modernas como NVIDIA V100, logrando un equilibrio óptimo entre velocidad y precisión.
        - **Comparación:**
            - YOLOv4 ofrece una mejora significativa en comparación con YOLOv3, tanto en mAP como en velocidad.
            - Compite directamente con otros modelos de detección como Faster R-CNN y EfficientDet, mostrando ventajas en tareas donde el tiempo real es crucial.
        
        ### **Conclusión**
        
        YOLOv4 se destaca como una solución ideal para la detección de objetos en escenarios donde se requiere precisión y eficiencia. Su diseño modular y las mejoras en data augmentation lo convierten en una herramienta poderosa para tareas prácticas de identificación de imágenes.

- **[A Machine Learning Based Intelligent Vision System for Autonomous Object Detection and Recognition](https://link.springer.com/article/10.1007/s10489-013-0461-5#citeas)**
    - **A Machine Learning Based Intelligent Vision System for Autonomous Object Detection and Recognition**
    - **Resumen:**
        
        Este artículo presenta un sistema de visión inteligente basado en **aprendizaje automático** para la detección y reconocimiento autónomo de objetos en entornos reales. El objetivo principal es desarrollar una solución robusta y eficiente que permita identificar y clasificar objetos en tiempo real, con aplicaciones potenciales en automatización, robótica y vehículos autónomos.
        
        ### **Modelos utilizados y herramientas adicionales**
        
        - **Modelos de detección de objetos:**
            - **YOLO (You Only Look Once):** Utilizado para realizar detecciones rápidas y precisas.
            - **Faster R-CNN:** Aplicado como modelo comparativo, conocido por su alta precisión en detección de objetos.
        - **Redes Convolucionales (CNNs):**
            - Backbone de los modelos para extraer características visuales a partir de las imágenes.
        - **Algoritmos adicionales:**
            - Métodos supervisados para mejorar la precisión en reconocimiento de clases específicas.
        
        ### **Técnicas de extracción de características**
        
        - **Mapas de características:** Generados por las CNNs para identificar patrones visuales relevantes.
        - **Bounding Boxes (cajas delimitadoras):** Utilizadas para localizar y delimitar los objetos detectados.
        - **Técnicas de preprocesamiento:** Incluyen normalización de imágenes y aumentos de datos (data augmentation) para robustecer el entrenamiento.
        
        ### **Dataset**
        
        - **Entrenamiento:** Se utilizó un conjunto de datos sintéticos y reales, combinando objetos de diferentes categorías, como vehículos, peatones y señales de tráfico.
        - **Validación:** Datasets estándar de detección de objetos, como COCO y Pascal VOC, para evaluar el rendimiento en entornos diversos.
        
        ### **Resultados y métricas**
        
        - **Precisión (Mean Average Precision, mAP):**
            - YOLO alcanzó una mAP de **78.6%**, destacando en velocidad y eficiencia.
            - Faster R-CNN logró una mAP de **82.3%**, con mejor precisión en clases complejas pero menor velocidad.
        - **Velocidad:**
            - YOLO procesó hasta **45 FPS (frames por segundo)**, siendo ideal para aplicaciones en tiempo real.
            - Faster R-CNN, aunque más preciso, se limitó a **7 FPS** debido a su complejidad computacional.
        
        ### **Comparación entre métodos**
        
        - **YOLO:** Es más adecuado para aplicaciones en tiempo real donde la velocidad es crucial.
        - **Faster R-CNN:** Recomendado para tareas que priorizan la precisión sobre la velocidad.
        - **Sistema híbrido:** El artículo sugiere que una combinación de ambos métodos podría optimizar el equilibrio entre velocidad y precisión.
        
        ### **Conclusión**
        
        El sistema propuesto combina modelos avanzados de detección de objetos con técnicas de aprendizaje automático para ofrecer una solución eficiente y precisa. Su enfoque permite adaptarse a diversas aplicaciones, desde sistemas de monitoreo hasta vehículos autónomos, demostrando la viabilidad de las tecnologías de visión artificial para entornos reales.

- **[Application of Deep Learning for Object Detection](https://www.sciencedirect.com/science/article/pii/S1877050918308767)**
    - **Application of Deep Learning for Object Detection**
    - **Resumen:**
        
        Este artículo analiza el uso de técnicas de aprendizaje profundo, específicamente redes neuronales convolucionales (CNNs), para tareas de detección de objetos en imágenes. Su objetivo principal es explorar y evaluar métodos que permitan localizar y clasificar objetos en entornos complejos, destacando sus aplicaciones en vigilancia, robótica y vehículos autónomos.
        
        ### **Modelos utilizados y herramientas adicionales**
        
        - **Modelos principales:**
            - **YOLO (You Only Look Once):** Optimizado para la detección rápida y precisa de múltiples objetos en tiempo real.
            - **Faster R-CNN:** Enfocado en la precisión, adecuado para contextos donde el tiempo de procesamiento no es crítico.
            - **SSD (Single Shot Multibox Detector):** Balance entre velocidad y precisión, recomendado para dispositivos con recursos limitados.
        - **Técnicas complementarias:**
            - Algoritmos para generar regiones propuestas, como en Faster R-CNN.
            - Métodos de supresión no máxima (NMS) para refinar la selección de objetos detectados.
        
        ### **Técnicas de extracción de características**
        
        - **Mapas de características:** Generados por CNNs para identificar patrones relevantes en las imágenes.
        - **Bounding Boxes:** Utilizadas para delimitar la ubicación de los objetos detectados.
        - **Data augmentation:** Aplicada para mejorar la robustez del modelo frente a variaciones como rotaciones, cambios de escala y ruido.
        
        ### **Dataset**
        
        - **COCO (Common Objects in Context):** Principal conjunto de datos empleado, con más de 80 categorías de objetos y millones de imágenes anotadas.
        - **Pascal VOC:** Utilizado para validación, ofreciendo un balance entre complejidad y diversidad.
        
        ### **Resultados y métricas**
        
        - **Mean Average Precision (mAP):**
            - **YOLO:** Alcanzó una mAP de **57.9%** en COCO, con procesamiento a **45 FPS**, ideal para aplicaciones en tiempo real.
            - **Faster R-CNN:** Obtuvo una mAP de **73.2%**, sacrificando velocidad por mayor precisión.
            - **SSD:** Se posicionó como un método intermedio con **mAP de 62.3%** y velocidad decente en dispositivos estándar.
        
        ### **Comparación entre métodos**
        
        - **YOLO:** Destacado por su velocidad, adecuado para aplicaciones en tiempo real como conducción autónoma y vigilancia.
        - **Faster R-CNN:** Más preciso, especialmente en escenarios complejos, pero con mayor costo computacional.
        - **SSD:** Balance entre velocidad y precisión, recomendado para dispositivos con recursos limitados.
        
        ### **Conclusión**
        
        El artículo concluye que los avances en aprendizaje profundo han transformado la detección de objetos, permitiendo aplicaciones prácticas en múltiples dominios. Los modelos evaluados ofrecen diferentes compromisos entre velocidad y precisión, adaptándose a necesidades específicas. También se destacan los desafíos futuros, como la detección en tiempo real en dispositivos de baja potencia y la mejora en escenarios de baja iluminación.
        
- **[Machine Learning Models in People Detection and Identification: A Literature Review
