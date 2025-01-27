from k10.Classifier_10k import Classifier_10K
from QA.QA_Classifier import Classifier_QA

from dataclasses import dataclass


@dataclass
class GlobalClassification:
    """
    A global classification manager that integrates two specific classifiers: 
    one for question-answer classifications and another for processing SEC Form 10-K reports.
    
    Attributes:
        model (str): The name of the language model used for classification.
        input_files_path (str): Path to the directory containing input CSV files.
        annotated_output_path (str): Path to the directory where annotated CSV files will be stored.
    """
    model: str
    input_files_path: str
    annotated_output_path: str

    def __post_init__(self):
        """
        Initializes the classifier instances after the dataclass attributes have been set.
        """
        self.classifier_qa = Classifier_QA(self.model, self.annotated_output_path)
        self.classifier_10k = Classifier_10K(
            annotated_csv_path=self.annotated_output_path,
            model=self.model
        )

    def process_qa_csvs(self):
        """
        Processes all CSV files for question-answer classification using the QA classifier.

        This method extracts and annotates question-answer pairs found in the specified input files directory. 
        It adds two columns to the CSV:
        1. A column indicating whether each intervention is classified as a question, answer, or procedure.
        2. A column specifying the question-answer pair to which the intervention belongs, if applicable.
        """
        self.classifier_qa.process_all_csvs(self.input_files_path)

    def process_10k_csvs(self):
        """
        Processes all CSV files for 10-K classification using the 10-K classifier.

        This method takes the CSV files generated in the previous step, where interventions are already classified into 
        question-answer pairs. Each intervention that belongs to a question-answer pair is further classified into one 
        of the following SEC 10-K report topics:

        Categories:
            [Business]: Outlines the company's operations, including its main products/services, target markets, market presence, and strategic objectives.
            [Risk Factors]: Identifies significant risks (market, regulatory, financial) that could adversely affect the company's stability or performance.
            [Selected Financial Data]: Highlights key financial metrics (revenue, net income, assets) over recent years, offering a high-level financial snapshot.
            [MD&A]: Analyzes financial results, explaining trends, challenges, and strategies, and provides management's perspective on future performance.
            [Financial Statements and Supplementary Data]: Contains audited financial statements (income statement, balance sheet, cash flow) and detailed notes adhering to accounting standards.
            [Other]: If none of the previous categories fits with the message of the text.

        For each observation:
                - The assigned category and the model's confidence score (1 - uncertainty) will be stored.
                - If the category is [Other], an explanation from the model will be included to justify why the intervention does not match any of the other categories.

        The resulting classifications are stored in a JSON file located in the same directory as the input CSV file.
        """
        self.classifier_10k.process_all_csvs()

    def process_all(self):
        """
        Processes all input CSV files using both the QA and 10-K classifiers sequentially.

        First, it processes the input files for question-answer classification using the QA classifier:
            - The resulting CSV files will have additional columns:
                1. A column indicating whether each intervention is classified as a question, answer, or procedure.
                2. A column specifying the question-answer pair to which the intervention belongs, if applicable.

        Next, it processes these expanded CSV files for SEC 10-K classification using the 10-K classifier:
            - Each question-answer pair (first question, then answer) is classified into one of six categories
            - The results are stored in a JSON file alongside the input CSV, detailing the classification and additional metadata for each detected question-answer pair.

        Returns:
            None: All results are saved as expanded CSV files and corresponding JSON files in the specified directories.
        """
        print("Procesando archivos para QA...")
        self.process_qa_csvs()
        print("Procesando archivos para 10K...")
        self.process_10k_csvs()
        print("Clasificación completada.")