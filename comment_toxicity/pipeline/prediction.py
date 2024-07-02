import numpy as np
import tensorflow as tf
from comment_toxicity import logger
import os
import pickle


STAGE_NAME = "Prediction stage"

class PredictionPipeline:
    def __init__(self,model_dir="artifacts/model_training/model",vectorizer_dir="artifacts/model_training/model"):
        self.model =tf.keras.models.load_model(f"{model_dir}/toxicity.h5")
        loaded_model = tf.keras.models.load_model(f"{model_dir}/vectorizer.keras")

        # The TextVectorization layer is now part of the loaded model
        # and can be accessed as needed
        self.vectorizer = loaded_model.layers[1]  # Assuming it's the second layer
        self.classes=["toxic","severe_toxic","obscene","threat","insult","identity_hate"]




    def score_comment(self,vectorized):
            result=self.model.predict(np.expand_dims(vectorized,0))
            text=''
            for idx,col in enumerate(self.classes):
                text+='{}: {}\n'.format(col,result[0][idx]>0.5)
            return text


    def model_prediction(self,text):
        try:
            input_text=self.vectorizer(text)
        except Exception as e:
            logger.exception(e)
            print("Invalid input")
            return None
        result=self.score_comment(input_text)
        return result
    
    def main(self,text):
        result=self.model_prediction(text)
        if result:
            print(result)
        else:
            print("Invalid input")
            
if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = PredictionPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e