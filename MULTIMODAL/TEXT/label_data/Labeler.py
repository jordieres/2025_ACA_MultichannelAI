import pandas as pd
import os
import re
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class Labeler:
    root_directory: str  # Path to get random samples
    output_file: str  # Path to save the labeled CSV file
    mode: str  # Either 'QA' or '10K'
    n_samples: int = 100
    label_mapping: Dict[int, str] = field(init=False)

    def __post_init__(self):
        """Configures label mapping and sample extraction function based on mode."""
        if self.mode == 'QA':
            self.label_mapping = {0: 'Question', 1: 'Answer', 2: 'Procedure'}
        elif self.mode == '10K':
            self.label_mapping = {
                0: 'Business',
                1: 'Risk Factors',
                2: 'Selected Financial Data',
                3: 'MD&A',
                4: 'Financial Statements and Supplementary Data'
            }
        else:
            raise ValueError("Invalid mode. Choose 'QA' or '10K'.")

    def get_random_sample(self) -> pd.DataFrame:
        """Fetches a random sample of observations with metadata based on the mode."""
        if self.mode == 'QA':
            return self.get_random_sample_with_metadata_QA()
        elif self.mode == '10K':
            return self.get_random_sample_with_metadata_10K()

    def get_random_sample_with_metadata_QA(self) -> pd.DataFrame:
        """Fetches a random sample of QA observations with metadata."""
        all_data = []
        for dirpath, _, filenames in os.walk(self.root_directory):
            for file in filenames:
                if file == 'transcript.csv':
                    file_path = os.path.join(dirpath, file)
                    try:
                        match = re.search(r'/companies/([^/]+)/(\d{4})/(Q[1-4])/transcript\.csv$', file_path)
                        if not match:
                            print(f"Metadata could not be extracted from: {file_path}")
                            continue

                        company, year, quarter = match.groups()
                        df = pd.read_csv(file_path)
                        df['company'] = company
                        df['year'] = year
                        df['quarter'] = quarter
                        all_data.append(df)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

        combined_data = pd.concat(all_data, ignore_index=True)
        return combined_data.sample(n=self.n_samples) if len(combined_data) > self.n_samples else combined_data

    def get_random_sample_with_metadata_10K(self) -> pd.DataFrame:
        """Fetches a random sample of 10K observations with metadata."""
        filtered_data = []
        for dirpath, _, filenames in os.walk(self.root_directory):
            for file in filenames:
                if file.endswith('.csv'):
                    file_path = os.path.join(dirpath, file)
                    try:
                        match = re.search(r'/([^/]+)/(\d{4})/(Q[1-4])\.csv$', file_path)
                        if not match:
                            print(f"Metadata could not be extracted from: {file_path}")
                            continue

                        company, year, quarter = match.groups()
                        df = pd.read_csv(file_path)
                        filtered_rows = df[df['Pair'].notna() & (df['Pair'] != "")].copy()
                        filtered_rows['company'] = company
                        filtered_rows['year'] = year
                        filtered_rows['quarter'] = quarter
                        filtered_data.append(filtered_rows)
                    except Exception as e:
                        print(f"Error processing file {file_path}: {e}")

        all_filtered_data = pd.concat(filtered_data, ignore_index=True)
        return all_filtered_data.sample(n=self.n_samples) if len(all_filtered_data) > self.n_samples else all_filtered_data

    def display_observation(self, index: int, text: str):
        """Displays the current observation and label options in the terminal."""
        os.system('cls' if os.name == 'nt' else 'clear')
        print(f"Observation {index}:")
        print(f"Text: {text}")
        print("Label options:")
        for key, value in self.label_mapping.items():
            print(f"  {key}: {value}")

    def label_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Iterates through the DataFrame rows and allows the user to label each row."""
        for index, row in df.iterrows():
            self.display_observation(index, row['text'])
            while True:
                try:
                    label_input = int(input("Enter the corresponding number for the label: "))
                    if label_input in self.label_mapping:
                        df.loc[index, 'label'] = self.label_mapping[label_input]
                        print(f"Label '{self.label_mapping[label_input]}' saved.\n")
                        break
                    else:
                        print("Invalid input. Please enter a valid number.")
                except ValueError:
                    print("Invalid input. Please enter an integer.")
        return df

    def save_data(self, df: pd.DataFrame):
        """Saves the labeled DataFrame to a CSV file."""
        df.to_csv(self.output_file, index=False)
        print(f"The labeled file has been saved as '{self.output_file}'.")

    def run(self):
        """Main method to execute the labeling process."""
        df = self.get_random_sample()
        df = self.label_data(df)
        self.save_data(df)






