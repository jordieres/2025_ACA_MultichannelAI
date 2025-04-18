from dataclasses import dataclass
from typing import List
import os

from .InterventionAnalyzer import InterventionAnalyzer
from .QuestionAnswerAnalizer import QAAnalyzer
from .CoherenceAnalyzer import CoherenceAnalyzer
from .MultiModalEmbeddings import MultimodalEmbeddings

from MULTIMODAL.AUDIO.AudioEmotionAnalzer import AudioEmotionAnalysis
from MULTIMODAL.TEXT.Analyze.TextEmotionAnalyzer import TextEmotionAnalyzer


@dataclass
class EnsembleInterventionAnalyzer:
    sec10k_model_names: List[str]
    qa_analyzer_model: str
    audio_model_name: str
    text_model_name: str
    NUM_EVALUATIONS: int = 5
    verbose: int = 1

    def __post_init__(self):
        self.classifiers = [
            InterventionAnalyzer(model=name, NUM_EVALUATIONS=self.NUM_EVALUATIONS)
            for name in self.sec10k_model_names]
        
        self.qa_analyzer = QAAnalyzer(model_name=self.qa_analyzer_model)
        self.coherence_analyzer = CoherenceAnalyzer(model_name=self.qa_analyzer_model)
        self.audio_emotion_analyzer = AudioEmotionAnalysis(model_name=self.audio_model_name)
        self.text_emotion_analyzer = TextEmotionAnalyzer(model_name=self.text_model_name)

    def ensemble_predict(self, text: str):
        results = []
        model_confidences = {}  # Para guardar: nombre_modelo -> (categoría, confianza)

        self._print_header("Predicciones individuales")

        for clf in self.classifiers:
            cat, conf = clf.get_pred(text)
            self._print(f"[{clf.model}] Predicted: {cat} | Confidence: {conf:.1f}%")
            results.append((cat, conf))
            model_confidences[clf.model] = {"Predicted_category": cat, "Confidence": round(conf, 2)}

        # Sumar las confianzas por categoría
        conf_sum = {}
        for cat, conf in results:
            conf_sum[cat] = conf_sum.get(cat, 0.0) + conf

        # Obtener la categoría con mayor confianza total
        best_cat, total_conf = max(conf_sum.items(), key=lambda x: x[1])
        avg_conf = round(total_conf / len(self.classifiers), 2)

        self._print_header("Resultado combinado")
        self._print(f"✅ Final prediction: {best_cat} | Combined confidence: {avg_conf:.1f}%")

        return best_cat, avg_conf, model_confidences

    def initialize_multimodal_model(self, output_csv_path: str, original_path: str):
        path_csv = output_csv_path
        path_json = os.path.join(original_path, "LEVEL_3.json")
        path_audio = os.path.join(original_path, "audio.mp3")  # o el nombre que uses

        if not all(map(os.path.exists, [path_csv, path_json, path_audio])):
            raise FileNotFoundError("Faltan uno o más archivos requeridos para embeddings")

        self.multimodal_embeddings = MultimodalEmbeddings(
            path_csv=path_csv,
            path_json=path_json,
            audio_file_path=path_audio,
            audio_emotion_analyzer=self.audio_emotion_analyzer,
            text_emotion_analyzer=self.text_emotion_analyzer
        )

        self._print("[INFO] Multimodal model initialized")

    def generate_structured_output(self) -> dict:
        df = self.multimodal_embeddings.sentences_df
        self.multimodal_embeddings.generar_embeddings()

        df = df[df['classification'].isin(['Monologue', 'Question', 'Answer'])].copy()

        result = {
            "monologue_interventions": {},
        }

        # MONOLOGUES
        monologues = df[df["classification"] == "Monologue"]
        for idx, group in monologues.groupby("intervention_id"):
            full_text = " ".join(group["text"])
            audio_embs = group["audio_embedding"].tolist()
            text_embs = group["text_embedding"].tolist()
            num_sent = len(audio_embs)

            result["monologue_interventions"][str(idx)] = {
                "text": full_text,
                "multimodal_embeddings": {
                    "num_sentences": num_sent,
                    "audio": audio_embs,
                    "text": text_embs
                }
            }

        # QA PAIRS
        qa_df = df[df["classification"].isin(["Question", "Answer"])]
        for pair_id, pair_group in qa_df.groupby("Pair"):
            if not isinstance(pair_id, str) or not pair_id.startswith("pair_"):
                continue  # 🛑 Saltar pares inválidos
            question_df = pair_group[pair_group["classification"] == "Question"]
            answer_df = pair_group[pair_group["classification"] == "Answer"]

            question_text = " ".join(question_df["text"])
            answer_text = " ".join(answer_df["text"])

            q_cat, q_conf, q_models = self.ensemble_predict(question_text)
            a_cat, a_conf, a_models = self.ensemble_predict(answer_text)

            try:
                evaluations = self.qa_analyzer.analize_qa(question_text, answer_text).get("evaluations", [])
            except:
                evaluations = []

            coherence_analyses = []
            for mono_id, monologue in result["monologue_interventions"].items():
                try:
                    coh = self.coherence_analyzer.analyze_coherence(monologue["text"], answer_text)
                    coh["monologue_index"] = int(mono_id)
                    coherence_analyses.append(coh)
                except:
                    pass

            result[pair_id] = {
                "Question": question_text,
                "Answer": answer_text,
                "question_classification": {
                    "Predicted_category": q_cat,
                    "Confidence": q_conf,
                    "Model_confidences": q_models
                },
                "answer_classification": {
                    "Predicted_category": a_cat,
                    "Confidence": a_conf,
                    "Model_confidences": a_models
                },
                "evaluations": evaluations,
                "coherence_analyses": coherence_analyses,
                "multimodal_embeddings": {
                    "question": {
                        "num_sentences": len(question_df["audio_embedding"].tolist()),
                        "audio": question_df["audio_embedding"].tolist(),
                        "text": question_df["text_embedding"].tolist()
                    },
                    "answer": {
                        "num_sentences": len(answer_df["audio_embedding"].tolist()),
                        "audio": answer_df["audio_embedding"].tolist(),
                        "text": answer_df["text_embedding"].tolist()
                    }
                }
            }

        return result
    
    def _print(self, *args, **kwargs):
        if self.verbose >= 1:
            print(*args, **kwargs)

    def _print_header(self, title):
        if self.verbose >= 1:
            print(f"\n{'='*10} {title} {'='*10}")
