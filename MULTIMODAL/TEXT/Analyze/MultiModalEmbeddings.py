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
            start_ms = int(start_time * 1000)
            end_ms = int((end_time + 0.25) * 1000)

            # Cortar el segmento
            segmento = self.full_audio[start_ms:end_ms]
            # display(Audio(segmento.export(format="mp3").read(), rate=44100))

            # Crear archivo temporal WAV
            temp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=True)
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