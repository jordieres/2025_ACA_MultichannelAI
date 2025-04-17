from dataclasses import dataclass

import pandas as pd
import numpy as np
import json
import re

from pydub import AudioSegment
import tempfile

def dummy_npwarn_decorator_factory():
  # https://stackoverflow.com/questions/77064579/module-numpy-has-no-attribute-no-nep50-warning SACADO DE AQUI
  def npwarn_decorator(x):
    return x
  return npwarn_decorator
np._no_nep50_warning = getattr(np, '_no_nep50_warning', dummy_npwarn_decorator_factory)


from MULTIMODAL.AUDIO.AudioEmotionAnalzer import AudioEmotionAnalysis
from MULTIMODAL.TEXT.Analyze.TextEmotionAnalyzer import TextEmotionAnalyzer

@dataclass
class MultimodalEmbeddings:
    path_csv: str
    path_json: str
    audio_file_path: str
    audio_emotion_analyzer: AudioEmotionAnalysis
    text_emotion_analyzer: TextEmotionAnalyzer
    

    def __post_init__(self):
        data_csv = pd.read_csv(self.path_csv)

        with open(self.path_json, 'r') as f:
            data_json = json.load(f)
        
        self.sentences_df = self.obtain_sentences_from_interventions(data_csv, data_json)

        # self.sentences_df.to_csv('SENTENCES_DF.csv')

        self.full_audio = AudioSegment.from_mp3(self.audio_file_path)


    def obtain_sentences_from_interventions(self, df_csv, data_json):
        frases_json = []
        for speaker in data_json.get("speakers", []):
            words = speaker.get("words", [])
            times = speaker.get("start_times", [])
            speaker_name = speaker.get("speaker_info", {}).get("name", "")

            frase = ""
            tiempos = []

            for palabra, tiempo in zip(words, times):
                frase += palabra + " "
                tiempos.append(tiempo)

                if re.match(r".*[\.!?]$", palabra):
                    frases_json.append({
                        "speaker": speaker_name,
                        "text": frase.strip(),
                        "start_time": tiempos[0],
                        "end_time": tiempos[-1]
                    })
                    frase = ""
                    tiempos = []

        df_json = pd.DataFrame(frases_json)

        frases_expandidas = []
        for _, row in df_csv.iterrows():
            frases = re.split(r'(?<=[\.!?])\s+', row["text"])
            for frase in frases:
                if frase.strip():
                    nueva_fila = row.to_dict()
                    nueva_fila["text"] = frase.strip()
                    nueva_fila["intervention_id"] = row["intervention_id"]
                    frases_expandidas.append(nueva_fila)

        df_expandido = pd.DataFrame(frases_expandidas)

        min_len = min(len(df_expandido), len(df_json))
        df_expandido = df_expandido.iloc[:min_len].copy()
        df_json = df_json.iloc[:min_len].copy()

        df_expandido["start_time"] = df_json["start_time"].values
        df_expandido["end_time"] = df_json["end_time"].values

        df_expandido["Pair"] = df_expandido["Pair"].fillna("Monologue")

        return df_expandido[df_expandido["classification"].isin(['Monologue', 'Question', 'Answer'])].reset_index(drop=True)
    

    def cortar_audio_temporal(self, start_time: int, end_time: int):
        """
        Recorta un archivo MP3 entre un rango de tiempo específico y devuelve un archivo temporal WAV.

        Args:
            input_file (str): Ruta del archivo MP3 de entrada.
            start_time (int): Tiempo inicial en segundos.
            end_time (int): Tiempo final en segundos.

        Returns:
            tempfile.NamedTemporaryFile: Archivo temporal WAV listo para usar.
        """
        try:
            # Convertir tiempos a milisegundos
            # start_ms = start_time * 1000
            # end_ms = end_time * 1000
            start_ms = int(start_time * 1000)
            end_ms = int((end_time + 0.25) * 1000)

            # Cortar el segmento
            segmento = self.full_audio[start_ms:end_ms]
            # display(Audio(segmento.export(format="mp3").read(), rate=44100))

            # Crear archivo temporal WAV
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            segmento.export(temp_wav.name, format="wav")

            return temp_wav

        except Exception as e:
            print(f"Error al procesar el archivo: {e}")
            return None
        
    
    def generar_embeddings(self) -> pd.DataFrame:
        audio_embs = []
        text_embs = []

        for _, row in self.sentences_df.iterrows():
            temp_file = self.cortar_audio_temporal(row['start_time'], row['end_time'])
            audio_emb = self.audio_emotion_analyzer.get_embeddings(temp_file.name)
            text_emb = self.text_emotion_analyzer.get_embeddings(row['text'])

            audio_embs.append(audio_emb.tolist() if hasattr(audio_emb, 'tolist') else audio_emb)
            text_embs.append(text_emb.tolist() if hasattr(text_emb, 'tolist') else text_emb)

        self.sentences_df["audio_embedding"] = audio_embs
        self.sentences_df["text_embedding"] = text_embs

    # def generar_embeddings(self) -> Dict[str, Dict[str, Dict[str, List[float]]]]:
    #     resultado = {}

    #     # for _, row in self.sentences_df.head(5).iterrows():
    #     for _, row in self.sentences_df.iterrows():
    #         pair = row['Pair']
    #         text = row['text']

    #         if pair not in resultado:
    #             resultado[pair] = {}

    #         if text not in resultado[pair]:
    #             temp_file = self.cortar_audio_temporal(row['start_time'], row['end_time'])

    #             audio_emb = self.audio_emotion_analyzer.get_embeddings(temp_file.name)
    #             text_emb = self.text_emotion_analyzer.get_embeddings(text)

    #             resultado[pair][text] = {
    #                 "audio": audio_emb.tolist() if hasattr(audio_emb, 'tolist') else audio_emb,
    #                 "text": text_emb.tolist() if hasattr(text_emb, 'tolist') else text_emb
    #             }

    #     return resultado

        # return self.sentences_df
    
    # def integrate_in_result(self, result: dict):
    #     embeddings_dict = self.generar_embeddings()
    #     for pair_key, frases in embeddings_dict.items():
    #         if pair_key in result['qa_pairs']:
    #             result['qa_pairs'][pair_key]['multimodal_embeddings'] = frases
    #         elif pair_key == 'Monologue':
    #             result['monologue_interventions_embeddings'] = frases
    #     return result



    # def integrate_in_result(self, result: dict):
    #     embeddings_dict = self.generar_embeddings()

    #     # Asegurar que los monólogos son dicts con clave 'text'
    #     if isinstance(next(iter(result["monologue_interventions"].values())), str):
    #         result["monologue_interventions"] = {
    #             idx: {"text": texto}
    #             for idx, texto in result["monologue_interventions"].items()
    #         }

    #     for pair_key, frases in embeddings_dict.items():
    #         # === QA PAIRS ===
    #         if pair_key in result['qa_pairs']:
    #             q_audio, q_text = [], []
    #             a_audio, a_text = [], []

    #             question_text = result['qa_pairs'][pair_key]['Question']
    #             answer_text = result['qa_pairs'][pair_key]['Answer']

    #             for emb in frases:
    #                 if emb['text'] in question_text:
    #                     q_audio.append(emb['audio'])
    #                     q_text.append(emb['text'])
    #                 elif emb['text'] in answer_text:
    #                     a_audio.append(emb['audio'])
    #                     a_text.append(emb['text'])

    #             result['qa_pairs'][pair_key]['question_embeddings'] = {
    #                 "audio": q_audio,
    #                 "text": q_text
    #             }
    #             result['qa_pairs'][pair_key]['answer_embeddings'] = {
    #                 "audio": a_audio,
    #                 "text": a_text
    #             }

    #         # === MONÓLOGOS ===
    #         elif pair_key == 'Monologue':
    #             for idx, monologue_data in result["monologue_interventions"].items():
    #                 audio_list, text_list = [], []
    #                 monologue_text = monologue_data["text"]

    #                 for emb in frases:
    #                     if emb['text'] in monologue_text:
    #                         audio_list.append(emb['audio'])
    #                         text_list.append(emb['text'])

    #                 result["monologue_interventions"][idx]["multimodal_embeddings"] = {
    #                     "audio": audio_list,
    #                     "text": text_list
    #                 }

    #     return result