from QA_Classifier import Classifier

model = 'llama3'
file_path = '/home/aacastro/mchai/companies/AAPL/2024/Q3/transcript.csv'
output_path = '/home/aacastro/mchai/annotated/'

classifier = Classifier(model, output_path)

file_path = '/home/aacastro/mchai/companies/'
classifier.process_all_csvs(file_path)