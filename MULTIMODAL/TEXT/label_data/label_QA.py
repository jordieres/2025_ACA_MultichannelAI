import pandas as pd
import os
import re
from dataclasses import dataclass, field
from typing import Dict

@dataclass
class Labeler_QA:
    root_directory: str  # Path to get random samples
    output_file: str  # Path to save the labeled CSV file
    n_samples: int = 100
    label_mapping: Dict[int, str] = field(default_factory=lambda: {
        0: 'Question',
        1: 'Answer',
        2: 'Procedure'
    })

    def load_data(self) -> pd.DataFrame:
        """Loads the CSV file into a pandas DataFrame."""
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"Input file '{self.input_file}' does not exist.")
        return pd.read_csv(self.input_file)

    def display_observation(self, index: int, text: str):
        """Displays the current observation and label options in the terminal."""
        # Clear terminal screen
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
                        print("Invalid input. Please enter a valid number (0, 1, 2).")
                except ValueError:
                    print("Invalid input. Please enter an integer.")
        return df

    def save_data(self, df: pd.DataFrame):
        """Saves the labeled DataFrame to a CSV file."""
        df.to_csv(self.output_file, index=False)
        print(f"The labeled file has been saved as '{self.output_file}'.")

    def get_random_sample_with_metadata_QA(self) -> pd.DataFrame:
            """Fetches a random sample of observations with metadata from the specified directory."""
            # List to store all observations with additional metadata
            all_data = []

            # Traverse the directory and its subdirectories
            for dirpath, _, filenames in os.walk(self.root_directory):
                for file in filenames:
                    if file == 'transcript.csv':  # Ensure only processing transcript.csv files
                        file_path = os.path.join(dirpath, file)
                        try:
                            # Extract company, year, and quarter from file_path using regex
                            match = re.search(r'/companies/([^/]+)/(\d{4})/(Q[1-4])/transcript\.csv$', file_path)
                            if not match:
                                print(f"Metadata could not be extracted from: {file_path}")
                                continue

                            company, year, quarter = match.groups()

                            # Read the CSV
                            df = pd.read_csv(file_path)

                            # Add metadata columns
                            df['company'] = company
                            df['year'] = year
                            df['quarter'] = quarter

                            all_data.append(df)
                        except Exception as e:
                            print(f"Error processing file {file_path}: {e}")

            # Concatenate all observations
            combined_data = pd.concat(all_data, ignore_index=True)

            # Randomly select n_samples
            if len(combined_data) <= self.n_samples:
                print(f"Warning: Fewer than {self.n_samples} observations available. Selecting {len(combined_data)}.")
                return combined_data
            else:
                return combined_data.sample(n=self.n_samples)

    def run(self):
        """Main method to execute the labeling process."""
        # Load data
        df = self.get_random_sample_with_metadata_QA()
        # Perform labeling
        df = self.label_data(df)
        # Save labeled data
        self.save_data(df)


if __name__ == "__main__":

    labeler = Labeler_QA(
        root_directory='/home/aacastro/mchai/companies/',
        output_file='/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/TEXT/label_data/labeled_gold_sample_QA.csv'
    )

    labeler.run()