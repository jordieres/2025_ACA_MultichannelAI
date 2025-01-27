from global_classification import GlobalClassification

model = 'llama3'
input_files_path = '/home/aacastro/mchai/companies/'
annotated_output_path = '/home/aacastro/mchai/annotated/'

global_classifier = GlobalClassification(model, input_files_path, annotated_output_path)
global_classifier.process_all()