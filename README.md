# Multimodal AI in Finance

This project implements a pipeline to analyze financial conferences by combining text, audio, and video processing. The main code is located in `src/multimodal_fin` and is organized into several modules.


## How to Use the System

1. **Installation**: First, create and activate a virtual environment (optional but recommended). Then, from the project root directory:
   ```bash
   cd 2025_ACA_MultichannelAI/
   pip install poetry
   poetry install
   ```

2. **Download SP500 Conference Data**: This will download a subset of earnings call data (transcripts and audio) for companies in the S&P 500 and store them under   the folder specified in your config.
   ```bash
   poetry run multimodal-fin download config/config.yaml
   ```

3. **Prepare the Conferences**: Create a CSV with the path to each conference (similar to `data/paths.csv`). Each folder must contain `transcript.csv` (derived from `LEVEL_3.json`, which contains interventions one by one with timestamps), `LEVEL_4.json` (which marks the separation between introduction and Q&A session), and the multimedia files.

4. **Run the Pipeline**: This includes textual classification, multimodal analysis, and generation of the enriched JSON.
   ```bash
   poetry run multimodal-fin process config/config.yaml --config-name default
   ```
   This will produce a CSV and an enriched JSON inside a `processed` folder next to each conference.

5. **Generate Embeddings**: For this step, pretrained weights of the proposed architecture are required. See `notebooks/train_encoders.ipynb`.
   
   For a single file:
   ```bash
   poetry run multimodal-fin embed config/config.yaml   --config-name default  --json-path /ruta/a/transcript.json
   ```

   For multiple files, create a CSV with a single column called `Paths` containing the paths to each `transcript.json`:
   ```bash
   poetry run multimodal-fin embed config/config.yaml   --config-name default  --json-csv data/json_paths.csv
   ```

## Expected Results
At the end, you will obtain:
- A CSV with classified and annotated interventions.
- A JSON containing multimodal embeddings and metadata (topic classification, coherence analysis, etc.).
- If the embeddings pipeline is used, vectors representing the entire conference for machine learning tasks.

![Embeddings Visualization](static/final_embeddings.png)
