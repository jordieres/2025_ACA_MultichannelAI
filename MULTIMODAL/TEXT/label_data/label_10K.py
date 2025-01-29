from Labeler import Labeler

# Classify 10K categories
labeler_10K = Labeler(
    root_directory='/home/aacastro/mchai/annotated/',
    output_file='/home/aacastro/Alejandro/ACA_MultichanelAI_2025/src/TEXT/label_data/labeled_gold_sample_10K.csv',
    mode='10K'
)
labeler_10K.run()
