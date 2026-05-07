class model_loader:
    def __init__(self):
        pass

    
    def __load_model(self):
        loaders = {
            'keras': self.__load_keras,
            'onnx':  self.__load_onnx,
            'pt':    self.__load_pytorch,
            'pth':   self.__load_pytorch,
        }
        return loaders[self.format]()

    def __load_keras(self):
        original_dense_init = keras.layers.Dense.__init__

        def make_patched_init(original):
            def patched_init(self, *args, **kwargs):
                known_kwargs = original.__code__.co_varnames[:original.__code__.co_argcount]
                unknown = [k for k in kwargs if k not in known_kwargs]
                for k in unknown:
                    kwargs.pop(k)
                original(self, *args, **kwargs)
            return patched_init

        layers_to_patch = [
            keras.layers.Dense,
            keras.layers.Conv2D,
            keras.layers.LSTM,
            keras.layers.GRU,
            keras.layers.Embedding,
            keras.layers.BatchNormalization,
        ]
        originals = {layer: layer.__init__ for layer in layers_to_patch}
        for layer in layers_to_patch:
            layer.__init__ = make_patched_init(originals[layer])

        try:
            model = keras.saving.load_model(self.file_path)
        except TypeError as e:
            raise RuntimeError(f"Failed to load Keras model: {e}")
        finally:
            for layer, original in originals.items():
                layer.__init__ = original

        return model