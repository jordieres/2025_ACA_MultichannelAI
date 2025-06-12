# Multimodal AI in Finance

Este proyecto implementa un pipeline para analizar conferencias financieras mezclando procesamiento de texto, audio y video. El código principal se encuentra en `src/multimodal_fin` y está organizado en varios módulos.

## Componentes principales

### Preprocessor
- Ubicación: `src/multimodal_fin/processors/Preprocessor.py`
- Divide la transcripción en secciones de "prepared remarks" y "q_a".
- Clasifica cada intervención usando un ensamble de modelos (`EnsembleInterventionClassifier`).
- Anota pares pregunta‑respuesta.
- Guarda el resultado como CSV.

### MultimodalProcessor
- Ubicación: `src/multimodal_fin/processors/MultiModalProcessor.py`
- Extrae embeddings de audio, texto y video a través de `EmbeddingsExtractor`.
- Enriquecce esos embeddings con información obtenida de LLMs mediante `MetadataEnricher` (clasificación de temas, análisis de QA y coherencia).
- Genera un archivo JSON con toda la metadata.

### ConferenceProcessor
- Ubicación: `src/multimodal_fin/processors/conference.py`
- Orquesta todo el flujo para cada conferencia: preprocesado, extracción de embeddings y enriquecimiento.
- Lee las rutas de un CSV y deja los resultados procesados en una carpeta `processed`.

### Embeddings Pipeline
- Ubicación: `src/multimodal_fin/embeddings`
- Permite crear representaciones vectoriales a partir del JSON enriquecido. Incluye modelos de nodos y de conferencias (`NodeEncoder` y `ConferenceEncoder`).

### CLI
- Ubicación: `src/multimodal_fin/cli.py`
- Expone dos comandos con [Typer](https://typer.tiangolo.com/):
  - `process`: ejecuta el pipeline completo (texto y multimodal) a partir de un archivo de configuración YAML.
  - `embed`: genera embeddings de un JSON previamente enriquecido.

## Configuración
El archivo `config/config.yaml` contiene un ejemplo de configuración. Cada sección bajo `conferences_processing` define:
- Rutas de entrada (`input_csv_path`).
- Listas de modelos para las distintas tareas (QA, monólogo, sec10k, etc.).
- Parámetros de configuración de los dos transformers para la extracción de embeddings.

También puede incluir un bloque `embeddings_pipeline` para los parámetros de los encoders cuando se utilizan los comandos de generación de embeddings.

## Cómo usar el sistema

1. **Instalación**: Ejecutar desde el directorio raíz del proyecto:
   ```bash
   cd 2025_ACA_MultichannelAI/
   pip install -e .
   ```

2. **Preparar las conferencias**: crear un CSV con la ruta de cada conferencia (similar a `data/paths.csv`). Cada carpeta debe contener `transcript.csv` (derivado de `LEVEL_3.json` que contiene las intervenciones una a una con timestamps), `LEVEL_4.json` (que marca la separación entre introducción y ronda de preguntas-respuestas) y los archivos multimedia.

3. **Ejecutar el pipeline**: Esto incluye clasificación textual, análisis multimodal y generación del JSON enriquecido.
   ```bash
   multimodal-fin process --config-file config/config.yaml --config-name default
   ```
   Esto produce un CSV y un JSON enriquecido dentro de una carpeta `processed` al lado de cada conferencia.

4. **Generar embeddings**: Para este paso será necesario proporcionar unos pesos entrenados de la arquitectura propuesta. Ver `notebooks/train_encoders.ipynb`
   
   Para un único archivo:
   ```bash
   multimodal-fin embed   --config-file config/config.yaml   --config-name default  --json-path /ruta/a/transcript.json
   ```

   Para varios archivos habrá que meter los path de los transcript.json en un csv con una única columna 'Paths'.
   ```bash
   multimodal-fin embed   --config-file config/config.yaml   --config-name default  --json-csv data/json_paths.csv
   ```

## Resultados esperados
Al finalizar se obtienen:
- Un CSV con las intervenciones clasificadas y anotadas.
- Un JSON con embeddings multimodales y metadata (clasificación temática, análisis de coherencia, etc.).
- Si se usa el pipeline de embeddings, vectores que representan la conferencia completa para tareas de aprendizaje automático.

![Visualización de embeddings](static/final_embeddings.png)