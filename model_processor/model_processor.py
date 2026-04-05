import tensorflow as tf
import keras

class model_processor:
    def __init__(self, file_path='../test/test_model.keras'): #TODO: update default path after initial testing
        self.model = self.__upload_model(file_path)
    
    def __upload_model(self, file_path: str):
        file_type = file_path.split('.')[-1]
        model = None
        if file_type == 'keras':
            model = keras.saving.load_model(file_path)
        #TODO: extend with more file types
        return model

  