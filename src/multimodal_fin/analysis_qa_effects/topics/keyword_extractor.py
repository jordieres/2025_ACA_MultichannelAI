from typing import List
import pandas as pd
from tqdm import tqdm

class KeywordExtractor:
    def __init__(self, kw_model, top_n: int = 5):
        self.kw_model = kw_model  # KeyBERT instanciado fuera
        self.top_n = top_n

    def _extract_keywords(self, text: str) -> List[str]:
        return [kw for kw, _ in self.kw_model.extract_keywords(text, top_n=self.top_n)]

    def add_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["keywords"] = df["qa_text_clean"].apply(self._extract_keywords)
        # tqdm.pandas(desc="Extracting QA keywords")
        # df["keywords"] = df["qa_text_clean"].progress_apply(self._extract_keywords)
        return df