from global_classification import GlobalClassification

model_QA = 'llama3'
model_10K = 'llama3'
input_files_path = '/home/aacastro/mchai/companies/'
annotated_output_path = '/home/aacastro/mchai/annotated/'

global_classifier = GlobalClassification(model_QA, model_10K, input_files_path, annotated_output_path)
global_classifier.process_all()