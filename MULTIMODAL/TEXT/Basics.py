from abc import ABC, abstractmethod
from dataclasses import dataclass
import pandas as pd
import ollama
from collections import Counter


@dataclass
class LLMClient:
    model: str

    def __post_init__(self):
        self.model = self._normalize_model_name(self.model)
        self._ensure_model()

    def _normalize_model_name(self, model_name: str) -> str:
        return model_name if ':' in model_name else f"{model_name}:latest"

    def _ensure_model(self):
        if self.model not in [m.model for m in ollama.list().models]:
            ollama.pull(self.model)
            print(f"Model downloaded: {self.model}")

    def chat(self, messages: list, schema=None) -> str:
        response = ollama.chat(model=self.model, messages=messages, format=schema) if schema \
                   else ollama.chat(model=self.model, messages=messages)
        return response.message.content
    

@dataclass
class BaseClassifier(ABC):
    llm: LLMClient

    @abstractmethod
    def classify_text(self, text: str) -> str:
        """
        Método abstracto que debe implementar la lógica de clasificación de texto.
        """
        pass

    def classify_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Clasifica cada entrada del DataFrame en base al contenido de una columna de texto.
        Añade una columna 'classification' con los resultados.
        """
        df['classification'] = df[text_column].apply(self.classify_text)
        return df  


class UncertaintyMixin:
    def get_result_and_uncertainty(self, predict_fn, text: str, n: int = 5):
        predictions = [predict_fn(text) for _ in range(n)]
        counter = Counter(predictions)
        top_cat, top_freq = counter.most_common(1)[0]
        
        # Calculamos confianza como porcentaje: (k / n) * 100
        confidence = round((top_freq / n) * 100, 2)
        
        return top_cat, confidence