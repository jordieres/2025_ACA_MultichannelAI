from Labeler import Labeler

# Classify Question-Answer pairs
labeler_QA = Labeler(
    root_directory='/home/aacastro/mchai/companies/',
    output_file='/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/TEXT/label_data/labeled_gold_sample_QA.csv',
    mode='QA'
)
labeler_QA.run()