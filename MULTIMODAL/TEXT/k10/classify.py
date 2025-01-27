from Classifier_10k import Classifier_10K

model = 'llama3'
NUM_EVALUATIONS = 10
input_files_path = '/home/aacastro/mchai/annotated'
classifier = Classifier_10K(annotated_csv_path=input_files_path, model=model, NUM_EVALUATIONS=NUM_EVALUATIONS)

classified_pairs = classifier.process_all_csvs()