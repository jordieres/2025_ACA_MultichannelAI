import ollama

import os
import json
import pandas as pd
from collections import Counter

from typing import Literal
from pydantic import BaseModel
from dataclasses import dataclass



class Category_10K(BaseModel):
  """
  A class created to define the output from the LLM.
  Attributes:
      category (str): The name of the category predicted by the LLM
  """
  category: Literal['Business', 'Risk Factors', 'Selected Financial Data', 'MD&A', 'Financial Statements and Supplementary Data', 'Other']


@dataclass
class Classifier_10K:
    """
    A classifier tailored to process and annotate text interventions extracted from financial reports,
    specifically aligning them with sections of the SEC Form 10-K.

    Attributes:
        annotated_csv_path (str): Path to the directory containing annotated CSV files.
        model (str): The name of the language model used for classification. Defaults to 'llama3'.
        NUM_EVALUATIONS (int): Number of classification evaluations for uncertainty calculation. Defaults to 10.
    """

    annotated_csv_path: str
    model: str = 'llama3'
    NUM_EVALUATIONS: int = 10


    def get_pairs_from_csv(self, df: pd.DataFrame) -> dict:
        """
        Extracts question-answer pairs from a DataFrame where classifications are 'Question' or 'Answer'.

        Args:
            df (pd.DataFrame): DataFrame containing text and classification columns.

        Returns:
            dict: A dictionary of question-answer pairs grouped by their pair identifier.
        """
        filtered_pairs = df[df['classification'].isin(['Question', 'Answer'])]

        result = {}
        grouped = filtered_pairs.groupby('Pair')

        for pair, group in grouped:
            question = group.loc[group['classification'] == 'Question', 'text'].values
            answer = group.loc[group['classification'] == 'Answer', 'text'].values

            result[pair] = {
                'Question': question[0] if len(question) > 0 else None,
                'Answer': answer[0] if len(answer) > 0 else None
            }

        return dict(sorted(result.items(), key=lambda pair: int(pair[0].split('_')[1])))
    
    def save_json_from_csv(self, file_path: str, classified_pairs: dict) -> None:
        """
        Saves classified question-answer pairs to a JSON file.

        Args:
            file_path (str): Path to the source CSV file.
            classified_pairs (dict): Classified pairs to save.
        """
        json_path = os.path.splitext(file_path)[0] + ".json"
        with open(json_path, 'w') as json_file:
            json.dump(classified_pairs, json_file, indent=4)
    
    def classify_text_10k(self, text: str):
        """
        Classifies a text snippet into a section of the SEC Form 10-K.

        Args:
            text (str): The text to classify.

        Returns:
            str: The classification result.
        """
        messages = [
            {
                'role': 'system',
                'content': 
                """
                You are an expert in financial reporting and the structure of the SEC Form 10-K.  You will be given a text excerpt from an earnings call or a conference where a company in the S&P 500 
                presents its results. Your task is to analyze the content and determine which section of the Form 10-K it primarily aligns with.

                [Business]: Outlines the company's operations, including its main products/services, target markets, market presence, and strategic objectives.
                [Risk Factors]: Identifies significant risks (market, regulatory, financial) that could adversely affect the company's stability or performance.
                [Selected Financial Data]: Highlights key financial metrics (revenue, net income, assets) over recent years, offering a high-level financial snapshot.
                [MD&A]: Analyzes financial results, explaining trends, challenges, and strategies, and provides management's perspective on future performance.
                [Financial Statements and Supplementary Data]: Contains audited financial statements (income statement, balance sheet, cash flow) and detailed notes adhering to accounting standards.   
                [Other]: If none of the previous categories fits with the message of the text. 
                """
            },
            {
                'role': 'user',
                'content': f"""    
                Here is the text of the intervention: "{text}"
                """
            }
        ]

        return json.loads(ollama.chat(model=self.model, messages=messages, format=Category_10K.model_json_schema()).message.content)['category']
        
    def explain_other_category(self, text: str):
        """
        Provides an explanation for why a text was classified as 'Other'.

        Args:
            text (str): The text to explain.

        Returns:
            str: The explanation provided by the model.
        """
        messages = [
            {
                'role': 'system',
                'content': 
                """
                You are an expert in financial reporting and the structure of the SEC Form 10-K. 
                Previously, a text excerpt was analyzed to classify it into one of the following categories:
                
                [Business]: Outlines the company's operations, including its main products/services, target markets, market presence, and strategic objectives.
                [Risk Factors]: Identifies significant risks (market, regulatory, financial) that could adversely affect the company's stability or performance.
                [Selected Financial Data]: Highlights key financial metrics (revenue, net income, assets) over recent years, offering a high-level financial snapshot.
                [MD&A]: Analyzes financial results, explaining trends, challenges, and strategies, and provides management's perspective on future performance.
                [Financial Statements and Supplementary Data]: Contains audited financial statements (income statement, balance sheet, cash flow) and detailed notes adhering to accounting standards. 
                
                The excerpt was classified as [Other], meaning it did not fit into any of the above categories. 

                Your task is to carefully analyze the text again and provide a detailed explanation of why it does not align with any of these categories. 
                """
            },
            {
                'role': 'user',
                'content': f"""
                Here is the text of the intervention: "{text}"
                """
            }
        ]

        return ollama.chat(model=self.model, messages=messages).message.content
    
    def get_result_and_uncertainty(self, predicted_categories: list):
        """
        Determines the most frequent classification and its uncertainty.

        Args:
            predicted_categories (list): List of predicted categories.

        Returns:
            tuple: The most common category and its confidence level.
        """
        category_counts = Counter(predicted_categories).most_common()
        
        most_common_category, most_common_frequency = category_counts[0]
        
        if len(category_counts) > 1:
            second_most_common_frequency = category_counts[1][1]
            uncertainty = (second_most_common_frequency / most_common_frequency)
        else:
            uncertainty = 0  # If there is not a second most voted category, the uncertainty is 0
        conficende = (1 - uncertainty) * 100

        return (most_common_category, conficende)
    
    def get_pred(self, text: str):
        """
        Repeatedly classifies a text to calculate uncertainty.

        Args:
            text (str): The text to classify.

        Returns:
            tuple: The most common category and its confidence level.
        """
        predicted_categories = [self.classify_text_10k(text) for _ in range(self.NUM_EVALUATIONS)]
        return self.get_result_and_uncertainty(predicted_categories)
    
    def process_text(self, text: str, pairs: dict, pair_key: str, q_a: str):
        """
        Processes and classifies a question or answer.

        Args:
            text (str): The text to classify.
            pairs (dict): Dictionary of question-answer pairs.
            pair_key (str): Key of the current pair.
            q_a (str): Indicates whether the text is a 'Question' or 'Answer'.

        Returns:
            dict: Updated pairs with classification results.
        """
        result = self.get_pred(text)
        print(f'Predicted category for {q_a}: {result[0]} | Confidence: {result[1]}')
        key = f"{q_a.lower()}_pred"
        pairs[pair_key][key] = {'Predicted_category': result[0], 'Confidence': result[1]}

        if result[0] == 'Other':
            explanation_key = f"why_other_{q_a.lower()}"
            pairs[pair_key][explanation_key] = self.explain_other_category(text)

        return pairs
    
    def classify_pairs(self, pairs: dict) -> None:
        """
        Classifies all question-answer pairs.

        Args:
            pairs (dict): Dictionary of question-answer pairs.

        Returns:
            dict: Updated pairs with classification results.
        """
        for pair_key, pair_value in pairs.items():
            question = pair_value.get("Question", "No question available")
            answer = pair_value.get("Answer", "No answer available")

            pairs = self.process_text(question, pairs, pair_key, 'Question')
            pairs = self.process_text(answer, pairs, pair_key, 'Answer')

            print('-'*50)
        return pairs

    def process_all_csvs(self) -> dict:
        """
        Processes all annotated CSV files in the specified directory.

        Saves the classification results as JSON files in the same directory.
        """
        for root, _, files in os.walk(self.annotated_csv_path):
            for file in files:
                if file.endswith('.csv'):  # Only process CSV files
                    file_path = os.path.join(root, file)
                    try:
                        pairs = self.get_pairs_from_csv(pd.read_csv(file_path))
                        classified_pairs = self.classify_pairs(pairs)
                        self.save_json_from_csv(file_path, classified_pairs)
                    except Exception as e:
                        print(f"Failed to process {file_path}: {e}")