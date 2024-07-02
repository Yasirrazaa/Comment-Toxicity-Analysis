from comment_toxicity import logger
import pandas as pd
import subprocess
import os  # Import the os module
import zipfile  # Import the zipfile module




STAGE_NAME = "Data Ingestion stage"

class DataIngestionPipeline:
    def __init__(self):
        self.data_dir = "artifacts/data_ingestion/data"  # Specify the directory for dataset download

    def setup_data_directory(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"Created data directory at {self.data_dir}")
        else:
            logger.info(f"Data directory {self.data_dir} already exists")

    
    def extract_zip_file(self, zip_file_path, extract_to):
        with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        logger.info(f"Extracted {zip_file_path} to {extract_to}")
        
        
    def download_dataset(self):
        command = f"kaggle competitions download -c jigsaw-toxic-comment-classification-challenge -p {self.data_dir}"
        if not os.path.exists(os.path.join(self.data_dir, "jigsaw-toxic-comment-classification-challenge.zip")):
            subprocess.run(command.split(), capture_output=False, text=False)
            logger.info("Dataset downloaded successfully.")
        else:
            logger.info("Dataset already exists.")
            
        zip_file_path = os.path.join(self.data_dir, "jigsaw-toxic-comment-classification-challenge.zip")
        self.extract_zip_file(zip_file_path, self.data_dir)
        train_path = os.path.join(self.data_dir, "train.csv.zip")
        test_path = os.path.join(self.data_dir, "test.csv.zip")
        test_labels_path = os.path.join(self.data_dir, "test_labels.csv.zip")
        self.extract_zip_file(test_labels_path, self.data_dir)
        self.extract_zip_file(test_path, self.data_dir)
        self.extract_zip_file(train_path, self.data_dir)
    
    def main(self):
        self.setup_data_directory()  # Ensure the data directory is set up
        self.download_dataset()  # Download the dataset

if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e