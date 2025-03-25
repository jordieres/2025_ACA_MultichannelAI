import pandas as pd
import time

from AudioEmotionAnalizer import AudioEmotionAnalysis
from ..ModelMetrics import ModelMetrics


ROWS_PER_EMOTION = 10

DATASET = 'CREMA-D'
# DATASET = 'RAVDESS'

results_df = pd.DataFrame(columns=["model_name", "elapsed_time", "accuracy", "precision", "recall", "f1_score"])
df = pd.read_csv(f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/AUDIO/{DATASET}.csv')
df['label'] = df['label'].str.lower()
label_mapping = {"calm": "neutral", "disgust": "disgusted", "fear": "fearful"}
df["label"] = df["label"].replace(label_mapping)
df_sampled = df.groupby("label").sample(n=ROWS_PER_EMOTION, random_state=10)
df = df_sampled



models = ['emotion2vec_plus_seed', 'emotion2vec_plus_base', 'emotion2vec_plus_large', 'emotion2vec_base_finetuned']
df_results = pd.DataFrame()

for model in models:
    model_results_df = pd.DataFrame()  # DataFrame específico por modelo

    print(f"Probando emotion2vec con modelo={model}")

    try:
        model_name = model
        results_path = f'/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/AUDIO/sim_results/{DATASET}/emo2vec/{model_name}/'

        analyzer = AudioEmotionAnalysis(
            mode= "emotion2vec", 
            model_name= f"iic/{model_name}", 
            embeddings_output_dir= results_path + 'embeddings/')

        evaluator = ModelMetrics(model_name=model_name, results_path=results_path, model=model_name)

        start_time = time.time()
        df = analyzer.classify_dataframe(df)
        elapsed_time = round(time.time() - start_time, 2)

        # Obtener métricas
        metrics = evaluator.get_results(df)
        
        # Crear DataFrame con los resultados de esta iteración
        iteration_results_df = pd.DataFrame([
            {"model": model_name, "elapsed_time": elapsed_time, **metrics}
        ])
        
        # Agregar a los DataFrames correspondientes
        df_results = pd.concat([df_results, iteration_results_df], ignore_index=True)

    except Exception as e:
        print(f"An error occurred with model {model_name}: {e}")
        continue
    
    print('-'*100)

# Guardar el DataFrame global al final
df_results.to_csv(f"/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/AUDIO/sim_results/{DATASET}/metrics_global_.csv", index=False)
evaluator.plot_f1_vs_time_all_models(df_results)