from comment_toxicity import logger
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision,Recall,CategoricalAccuracy
from tensorflow.keras.layers import TextVectorization,Embedding,Bidirectional,GRU,Dense,BatchNormalization
import numpy as np
import pickle



STAGE_NAME = "Model Training stage"


class TrainingPipeline:
    def __init__(self,data_dir="artifacts/data_ingestion/data",model_dir="artifacts/model_training/model",epochs=5):
        self.data_dir = data_dir  # Specify the directory for dataset download
        self.model_dir = model_dir  # Specify the directory for model saving
        self.MAX_FEATURES=200000
        self.epochs=epochs

    def data_preparation(self):
        df = pd.read_csv(f"{self.data_dir}/train.csv")
        train_df=df['comment_text']
        test_df = df[df.columns[2:]].values
        test_labels_df = pd.read_csv(f"{self.data_dir}/test_labels.csv")
        logger.info(f"Train data shape: {train_df.shape}")
        logger.info(f"Test data shape: {test_df.shape}")
        logger.info(f"Test labels shape: {test_labels_df.shape}")

        
        vectorizer=TextVectorization(max_tokens=self.MAX_FEATURES,output_sequence_length=1800,output_mode='int')

        vectorizer.adapt(train_df.values)
        vectorized_text=vectorizer(train_df.values)
        
        dataset=tf.data.Dataset.from_tensor_slices((vectorized_text,test_df)) 
        dataset=dataset.cache()
        dataset=dataset.shuffle(160000)
        dataset=dataset.batch(64)
        dataset=dataset.prefetch(8)
        
        train=dataset.take(int(len(dataset)*.7))
        val=dataset.skip(int(len(dataset)*.7)).take(int(len(dataset)*.2))
        test=dataset.skip(int(len(dataset)*.9)).take(int(len(dataset)*.1))
        # Wrap the vectorizer in a Keras model
        inputs = tf.keras.Input(shape=(1,), dtype=tf.string)
        outputs = vectorizer(inputs)
        vectorizer = tf.keras.Model(inputs, outputs)

        # Save the model
        vectorizer.save(f"{self.model_dir}/vectorizer.keras")

        return train,val,test
    
    def model_creation(self):
        model = Sequential()
        # Create the embedding layer 
        model.add(Embedding(self.MAX_FEATURES+1, 128))
        # Bidirectional LSTM Layer
        model.add(Bidirectional(GRU(128, activation='tanh')))
        model.add(BatchNormalization())
        # Feature extractor Fully connected layers
        model.add(Dense(128, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())

        model.add(Dense(128, activation='relu'))
        # Final layer 
        model.add(Dense(6, activation='sigmoid'))
        model.save(f"{self.model_dir}/base_model.keras")
        return model
    
    def model_training(self,train,val):
        model=self.model_creation()
        model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['CategoricalAccuracy'])
        model.fit(train,validation_data=val,epochs=self.epochs)
        model.save(f"{self.model_dir}/trained.keras")
    
    def model_evaluation(self,test,model):
        pre=Precision()
        re=Recall()
        acc=CategoricalAccuracy()
        
        
        for batch in test.as_numpy_iterator():
            x_true,y_true=batch
            yhat=model.predict(x_true)
            y_true=y_true.flatten()
            yhat=yhat.flatten()
            pre.update_state(y_true,yhat)
            re.update_state(y_true,yhat)
            acc.update_state(y_true,yhat)
        logger.info(f"Precision: {pre.result().numpy()}")
        logger.info(f"Recall: {re.result().numpy()}")
        logger.info(f"Accuracy: {acc.result().numpy()}")
        
        
        
    
        
        
        
    def main(self):
        train,val,test=self.data_preparation()
        self.model_training(train,val)
        self.model_evaluation(test)
        

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = TrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e