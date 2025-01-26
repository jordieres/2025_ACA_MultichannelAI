import os
import json
import pandas as pd

import ollama

from dataclasses import dataclass
from pydantic import BaseModel
from typing import Literal


class Category_QA(BaseModel):
  """
  A class created to define the output for the LLM.
  Attributes:
      category (str): The name of the category predicted by the LLM
  """
  category: Literal['Question', 'Answer', 'Procedure']


@dataclass
class Classifier:
    """
    A classifier designed to process and annotate text interventions from meetings or conferences.

    Attributes:
        model (str): The name of the language model used for classification. Defaults to 'llama3'.
        output_path (str): Path where the resulting CSV file from `expand_df` will be saved.
    """

    model: str = 'llama3'
    output_path: str = 'output'

    def __post_init__(self) -> None:
        """
        Normalizes the model name and ensures the specified model is downloaded.
        If the model is not available locally, it will be downloaded using the `ollama` library.
        """
        self.model = self._normalize_model_name(self.model)
        if not self._is_model_downloaded(self.model):
            ollama.pull(self.model)

    def _normalize_model_name(self, model_name: str) -> str:
        """
        Appends ':latest' to the model name if no version tag is specified.

        Args:
            model_name (str): The name of the model.

        Returns:
            str: The normalized model name.
        """
        return model_name if ':' in model_name else f"{model_name}:latest"

    def _is_model_downloaded(self, model_name: str) -> bool:
        """
        Checks if the specified model is already downloaded.

        Args:
            model_name (str): The name of the model.

        Returns:
            bool: True if the model is downloaded, False otherwise.
        """
        return model_name in (model.model for model in ollama.list().models)

    def classify_text(self, text: str) -> str:
        """
        Classifies a given text into one of three categories: "Question," "Answer," or "Procedure."

        Args:
            text (str): The text of the intervention.

        Returns:
            str: The classification result.
        """
        messages = [
            {
                'role': 'system',
                'content': """
                You are a model designed to classify interventions in meetings or conferences into three categories:
                [Question]: If the intervention has an interrogative tone or seeks information.
                [Answer]: If the intervention provides information or responds to a previous question.
                [Procedure]: If the intervention is part of the meeting protocol, such as acknowledgments, moderation steps, or phrases without substantial informational content.
                """
            },
            {
                'role': 'user',
                'content': f"""
                Here is the text of the intervention: "{text}"
                """
            }
        ]

        result = json.loads(ollama.chat(model=self.model, messages=messages, format=Category_QA.model_json_schema()).message.content)['category']
    
        # Validate the result
        if result in {"Question", "Answer", "Procedure"}:
            return result
        else:
            print(f"Unexpected classification result. Retrying...")
            return self.classify_text(text)  # Retry with the same text

    def classify_interventions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifies the 'text' column in a DataFrame and adds a 'classification' column.

        Args:
            df (pd.DataFrame): A DataFrame containing a column named 'text'.

        Returns:
            pd.DataFrame: The input DataFrame with an additional 'classification' column.
        """
        return df.assign(classification=[self.classify_text(text) for text in df['text']])

    def annotate_question_answer_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Annotates question-answer pairs in a DataFrame based on the 'classification' column.

        Args:
            df (pd.DataFrame): A DataFrame containing a 'classification' column.

        Returns:
            pd.DataFrame: The input DataFrame with additional 'Pair' and 'Type' columns.
        """
        pair_id = 1
        current_question_row = None
        pairs = []

        for index, row in df.iterrows():
            if row['classification'] == "Question":
                current_question_row = index
                pairs.append(None)
            elif row['classification'] == "Answer" and current_question_row is not None:
                pairs[current_question_row] = f"pair_{pair_id}"
                pairs.append(f"pair_{pair_id}")
                pair_id += 1
                current_question_row = None
            else:
                pairs.append(None)

        df['Pair'] = pairs

        pair_counts = df['Pair'].value_counts(dropna=True)
        invalid_pairs = pair_counts[pair_counts != 2]

        if not invalid_pairs.empty:
            raise ValueError(f"Error: The following pairs do not have exactly 2 observations:\n{invalid_pairs.to_dict()}")

        return df
    
    def save_to_csv(self, df: pd.DataFrame, output_path: str) -> None:
        """
        Saves the given DataFrame to the specified output path in CSV format.

        Args:
            df (pd.DataFrame): The DataFrame to be saved.

        Returns:
            None
        """
        try:
            df.to_csv(output_path, index=False)
        except Exception as e:
            raise ValueError(f"Failed to save DataFrame to {output_path}: {e}")
        
    def _generate_output_path(self, file_path: str) -> str:
        """
        Generates a dynamic output path based on the input file path.

        Args:
            file_path (str): The path to the input CSV file.

        Returns:
            str: The dynamically generated output path for the CSV file.
        """
        parts = file_path.strip('/').split('/')
        company = parts[-4]
        year = parts[-3]
        filename = parts[-2] + '.csv'

        output_dir = os.path.join(self.output_path, company, year)
        os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists

        return os.path.join(output_dir, filename)

    def expand_df(self, file_path: str) -> pd.DataFrame:
        """
        Main method. Loads a CSV file, classifies the text interventions, annotates question-answer pairs, and loads the resulting csv file.

        Args:
            file_path (str): The path to the CSV file.

        Returns:
            pd.DataFrame: A DataFrame with classified and annotated data.
        """
        output_path = self._generate_output_path(file_path)
        df = self.annotate_question_answer_pairs(self.classify_interventions(pd.read_csv(file_path)))
        self.save_to_csv(df, output_path)
        return df
    
    def process_all_csvs(self, input_directory: str) -> None:
        """
        Processes all CSV files in the given directory and its subdirectories.
        Each processed file is saved in the dynamically generated output path.

        Args:
            input_directory (str): The base directory containing the CSV files to process.

        Returns:
            None
        """
        for root, _, files in os.walk(input_directory):
            for file in files:
                if file.endswith('.csv'):  # Only process CSV files
                    file_path = os.path.join(root, file)
                    try:
                        self.expand_df(file_path)
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")