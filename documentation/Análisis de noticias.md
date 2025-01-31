# Análisis de prensa

Objetivo: Analizar artículos de noticias para categorizarlos y hallar polaridad en los mismos utilizando técnicas de Machine Learning

## Posibles soluciones

- Clasificación supervisada: Entrenar modelos como Naive Bayes, SVM, o Redes Neuronales en datasets etiquetados para categorizar las noticias.
- Análisis de sentimientos: Utilizar modelos como BERT o LSTM para determinar la polaridad (positiva, negativa o neutral) de los artículos.
- Clustering no supervisado: Usar algoritmos como K-Means o DBSCAN para agrupar noticias por temas relacionados.
- Modelos preentrenados y fine-tuning: Ajustar transformers como GPT o RoBERTa a tareas específicas de clasificación de noticias y análisis de polaridad.

## Papers encontrados

### A Systematic Review of NLP Methods for Sentiment Classification of Online News Articles:

- Link: [Systematic Review on NLP Methods for Sentiment Analysis in Online News Articles](https://ieeexplore.ieee.org/document/10308056)
- **Nombre**: A Systematic Review of NLP Methods for Sentiment Classification of Online News Articles
- **Resumen**: Este estudio revisa sistemáticamente los métodos de Procesamiento de Lenguaje Natural (NLP) aplicados al análisis de sentimientos en noticias. Examina técnicas tradicionales y recientes, destacando el uso de Transformers como BERT y GPT, que superan en precisión a los modelos clásicos. Utiliza datasets etiquetados como el **Financial PhraseBank** y el **SemEval**.
- **Modelos utilizados**: Naive Bayes, SVM, LSTM, BERT.
- **Notas importantes**: Aborda retos como el sesgo en los modelos y la interpretación de resultados, proponiendo soluciones como el ajuste fino y el aprendizaje justo. Este trabajo identifica que los modelos como BERT tienen una ventaja notable en la comprensión contextual avanzada, mientras que los enfoques tradicionales como Naive Bayes son más rápidos pero menos precisos en datasets complejos. Además, enfatiza la importancia de la evaluación continua para mitigar sesgos, especialmente en dominios sensibles como las noticias.

### Machine Learning Application for News Text Classification:

- **Link**: [Machine Learning Application for News Text Classification](https://ieeexplore.ieee.org/document/10048856)
- **Nombre**: Machine Learning Application for News Text Classification
- **Resumen**: Este trabajo explora la clasificación de textos de noticias utilizando algoritmos de Machine Learning como SVM, Random Forest y BERT. Utiliza datasets como **20 Newsgroups** y el **BBC News Dataset**. El estudio evalúa el impacto del preprocesamiento y la selección de características en el rendimiento de los modelos.
- **Modelos utilizados**: SVM, Random Forest, BERT.
- **Notas importantes**: Se destaca la importancia del ajuste de hiperparámetros y el preprocesamiento de datos para mejorar los resultados de clasificación. Aunque los modelos más avanzados como BERT logran alta precisión, también presentan un costo computacional significativo. Los modelos tradicionales como SVM y Random Forest son efectivos en términos de interpretabilidad y tiempo de entrenamiento, pero tienen limitaciones en la comprensión semántica avanzada que los modelos como BERT abordan con éxito. Este último destaca en la identificación de matices complejos, aunque requiere recursos computacionales elevados y un ajuste fino cuidadoso.

### Fake News Detection Using Deep Learning:

- **Link**: [Deep Learning for Fake News Detection](https://arxiv.org/abs/1708.01967)
- **Nombre**: Fake News Detection Using Deep Learning
- **Resumen**: Este estudio se centra en el análisis de noticias falsas mediante redes neuronales profundas. Se utilizan CNNs para detectar patrones textuales específicos asociados con desinformación. Utiliza el **LIAR Dataset** y el **Fake News Challenge Dataset** para entrenar y evaluar los modelos.
- **Modelos utilizados**: Redes Convolucionales (CNN), Word2Vec.
- **Notas importantes**: Incluye una comparación entre enfoques supervisados y no supervisados para la detección de contenido engañoso. Las CNNs son particularmente efectivas en identificar patrones específicos en frases o palabras claves que son indicativos de noticias falsas. Sin embargo, su rendimiento puede disminuir en casos donde los patrones no sean tan evidentes o cuando se enfrentan a datos no estructurados. El uso de Word2Vec para la representación semántica mejora la capacidad del modelo para generalizar, pero depende de un preprocesamiento robusto.

Sentiment Analysis in News Media Using Machine Learning Techniques:

- **Link**: [Sentiment Analysis in News Media Using Machine Learning](https://ieeexplore.ieee.org/document/8636124)
- **Nombre**: Sentiment Analysis in News Media Using Machine Learning Techniques
- **Resumen**: Este trabajo analiza sentimientos en noticias mediante Naive Bayes y Redes Neuronales. Utiliza una combinación de features de texto como TF-IDF y embeddings preentrenados. Los datasets utilizados incluyen el **IMDB Reviews Dataset** y un conjunto específico etiquetado manualmente de noticias políticas.
- **Modelos utilizados**: Naive Bayes, Redes Neuronales, Embeddings.
- **Notas importantes**: Examina la eficacia de diferentes técnicas de feature extraction y su impacto en la precisión de la clasificación. La combinación de TF-IDF con embeddings preentrenados proporciona una base sólida para modelos tanto simples como complejos. Mientras que Naive Bayes muestra eficiencia en términos de tiempo de entrenamiento, las Redes Neuronales destacan en datasets más grandes y complejos al capturar relaciones no lineales. Sin embargo, esto viene a costa de mayor tiempo y recursos computacionales.

Tabla resumen:

| Modelo                   | Dataset utilizado                         | Ventajas                                   | Desventajas                                | Resultados                                      |
| ------------------------ | ----------------------------------------- | ------------------------------------------ | ------------------------------------------ | ----------------------------------------------- |
| BERT                     | Financial PhraseBank, SemEval             | Excelente comprensión contextual           | Alto costo computacional                   | Alta precisión en análisis de sentimiento       |
| Naive Bayes              | IMDB Reviews Dataset, etiquetado manual   | Rápido, eficiente                          | Bajo rendimiento en datos complejos        | Moderada precisión en clasificación             |
| Redes Neuronales         | IMDB Reviews Dataset, etiquetado manual   | Captura relaciones no lineales             | Requiere más recursos                      | Buen rendimiento con datos complejos            |
| CNN + Word2Vec           | LIAR Dataset, Fake News Challenge Dataset | Identifica patrones específicos            | Limitado con datos no estructurados        | Eficaz en detección de noticias falsas          |
| SVM, Random Forest, BERT | 20 Newsgroups, BBC News Dataset           | Interpretabilidad, alta precisión con BERT | Requiere ajuste y recursos computacionales | Versátil y robusto para tareas de clasificación |

