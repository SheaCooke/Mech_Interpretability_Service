import numpy as np
import keras
import tensorflow as tf
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from typing import Optional
import pandas as pd
import sys
import types


class Model_Processor:
    
#TODO: cosine distance needs to be from 0 to 2
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.format = self.__detect_format()
        self.model = self.__load_model()
        self.model_data = self.__extract_model_data()


    SUPPORTED_DATASET_FORMATS = ['csv', 'npz']
    SUPPORTED_FORMATS = ['keras'] #['keras', 'onnx', 'pt', 'pth']

    def __detect_format(self) -> str:
        ext = self.file_path.split('.')[-1].lower()
        if ext not in self.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: .{ext}. Supported formats: {self.SUPPORTED_FORMATS}")
        return ext

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

    def __load_onnx(self):
        model = onnx.load(self.file_path)
        onnx.checker.check_model(model)
        return model

    def __load_pytorch(self):
        # Create a permissive dynamic module that returns a generic nn.Module
        # subclass for any attribute access. This satisfies the unpickler when
        # the saved model references a class defined in __main__ of the
        # training script (e.g. MNISTModel), which does not exist in the
        # backend's __main__.
        class _AnyModule(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()
            def forward(self, x):
                return x

        class _PermissiveModule(types.ModuleType):
            def __getattr__(self, name):
                # Return a fresh subclass for every unknown attribute so the
                # unpickler can instantiate whatever class the file references
                return type(name, (_AnyModule,), {})

        # Temporarily inject the permissive module as __main__ so pickle can
        # resolve the class reference, then restore the original __main__
        original_main = sys.modules.get('__main__')
        sys.modules['__main__'] = _PermissiveModule('__main__')
        try:
            # First attempt: weights_only=True (safe, works for state dicts
            # and models saved with torch.save on PyTorch < 2.6)
            model = torch.load(self.file_path, map_location=torch.device('cpu'), weights_only=True)
        except Exception:
            # Second attempt: weights_only=False required for full model objects
            # (e.g. torch.save(model, path) where model is a custom nn.Module).
            # This is safe here because the file was explicitly uploaded by the user.
            model = torch.load(self.file_path, map_location=torch.device('cpu'), weights_only=False)

        if isinstance(model, dict):  # state_dict only, no architecture
            raise ValueError(
                "PyTorch file contains only a state_dict, not a full model. "
                "Save with torch.save(model, path) rather than torch.save(model.state_dict(), path)."
            )
        model.eval()
        return model


    def __extract_model_data(self) -> dict:
        extractors = {
            'keras': self.__extract_keras_model_data,
            'onnx':  self.__extract_onnx_model_data,
            'pt':    self.__extract_pytorch_model_data,
            'pth':   self.__extract_pytorch_model_data,
        }
        return extractors[self.format]()

    def __extract_keras_model_data(self) -> dict:
        layers = []
        for layer in self.model.layers:
            layer_weights = layer.get_weights()
            layer_data = {
                'name':               layer.name,
                'type':               type(layer).__name__,
                'trainable':          layer.trainable,
                'input_shape':        list(layer.input_shape) if hasattr(layer, 'input_shape') else None,
                'output_shape':       list(layer.output_shape) if hasattr(layer, 'output_shape') else None,
                'activation':         layer.activation.__name__ if hasattr(layer, 'activation') else None,
                'num_neurons':        layer.units if hasattr(layer, 'units') else None,
                'weight_shape':       list(layer_weights[0].shape) if layer_weights else None,
                'bias_shape':         list(layer_weights[1].shape) if len(layer_weights) > 1 else None,
                'num_weights':        layer_weights[0].size if layer_weights else 0,
                'num_biases':         layer_weights[1].size if len(layer_weights) > 1 else 0,
                'relevant_inference': not isinstance(layer, (keras.layers.Dropout,)),
            }
            layers.append(layer_data)

        return {
            'format':               'keras',
            'total_params':         self.model.count_params(),
            'trainable_params':     sum(tf.size(w).numpy() for w in self.model.trainable_weights),
            'non_trainable_params': sum(tf.size(w).numpy() for w in self.model.non_trainable_weights),
            'num_layers':           len(self.model.layers),
            'input_shape':          list(self.model.input_shape),
            'output_shape':         list(self.model.output_shape),
            'layers':               layers,
        }

    def __extract_onnx_model_data(self) -> dict:
        graph = self.model.graph
        layers = []
        for node in graph.node:
            layer_data = {
                'name':   node.name,
                'type':   node.op_type,
                'inputs': list(node.input),
                'outputs': list(node.output),
            }
            layers.append(layer_data)

        # Extract input/output shapes from graph
        input_shape = [dim.dim_value for dim in graph.input[0].type.tensor_type.shape.dim]
        output_shape = [dim.dim_value for dim in graph.output[0].type.tensor_type.shape.dim]

        # Count parameters from initializers (stored weights)
        total_params = sum(
            np.prod(initializer.dims)
            for initializer in graph.initializer
        )

        return {
            'format':       'onnx',
            'ir_version':   self.model.ir_version,
            'opset_version': self.model.opset_import[0].version,
            'total_params': int(total_params),
            'num_layers':   len(graph.node),
            'input_shape':  input_shape,
            'output_shape': output_shape,
            'layers':       layers,
        }

    def __extract_pytorch_model_data(self) -> dict:
        layers = []
        total_params = 0
        trainable_params = 0

        for name, module in self.model.named_modules():
            if name == '':  # skip root module
                continue
            params = list(module.parameters(recurse=False))
            num_params = sum(p.numel() for p in params)
            num_trainable = sum(p.numel() for p in params if p.requires_grad)
            total_params += num_params
            trainable_params += num_trainable

            layer_data = {
                'name':               name,
                'type':               type(module).__name__,
                'trainable':          any(p.requires_grad for p in params) if params else False,
                'num_params':         num_params,
                'activation':         None,  # PyTorch activations are often standalone modules
                'num_neurons':        module.out_features if hasattr(module, 'out_features') else None,
                'weight_shape':       list(module.weight.shape) if hasattr(module, 'weight') and module.weight is not None else None,
                'bias_shape':         list(module.bias.shape) if hasattr(module, 'bias') and module.bias is not None else None,
                'relevant_inference': not isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)),
            }

            # Capture activation function if it's a standalone module
            if isinstance(module, (nn.ReLU, nn.Sigmoid, nn.Tanh, nn.Softmax,
                                   nn.LeakyReLU, nn.ELU, nn.GELU, nn.SELU)):
                layer_data['activation'] = type(module).__name__

            layers.append(layer_data)

        return {
            'format':               'pytorch',
            'total_params':         total_params,
            'trainable_params':     trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'num_layers':           len(layers),
            'layers':               layers,
        }


    def run_inference(self, test_data: list[dict]) -> list[dict]:
        """
        test_data: list of dicts with keys 'id', 'input', and optionally 'label'
        e.g. [{'id': 'img_001', 'input': np.array(...), 'label': 7}, ...]
        returns the same list with 'activations' and 'predicted' added to each record
        """
        runners = {
            'keras': self.__run_keras_inference,
            'onnx':  self.__run_onnx_inference,
            'pt':    self.__run_pytorch_inference,
            'pth':   self.__run_pytorch_inference,
        }
        return runners[self.format](test_data)

    def __run_keras_inference(self, test_data: list[dict]) -> list[dict]:
        # Build activation model - outputs every layer's activations
        activation_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[layer.output for layer in self.model.layers]
        )

        results = []
        for record in test_data:
            input_data = np.expand_dims(record['input'], axis=0)  # add batch dimension
            layer_activations = activation_model.predict(input_data, verbose=0)

            # Concatenate all layer activations into a single flat 1D vector per record.
            # This allows efficient cosine distance computation across many records
            # using scipy.spatial.distance.cdist(activation_matrix, activation_matrix, 'cosine')
            # where activation_matrix is a 2D array of shape (num_records, total_activation_size)
            activations = np.concatenate([
                layer_activation[0].flatten()
                for layer_activation in layer_activations
            ])

            layer_names = [layer.name for layer in self.model.layers]

            per_layer = {
                name: la[0].tolist()
                for name, la in zip(layer_names, layer_activations)
            }

            results.append({
                'id':          record['id'], #TODO: id wont always be a column
                'input':       record['input'].tolist(),
                'label':       record.get('label'),
                'predicted':   int(np.argmax(layer_activations[-1][0])),
                'correct':     int(np.argmax(layer_activations[-1][0])) == record.get('label'),
                'activations': activations,
                'layer_activations': per_layer,
            })

        return results

    def __run_onnx_inference(self, test_data: list[dict]) -> list[dict]:
        # Clone model and register every intermediate node output
        model_copy = onnx.ModelProto()
        model_copy.CopyFrom(self.model)
        existing_outputs = {o.name for o in model_copy.graph.output}
        for node in model_copy.graph.node:
            for output_name in node.output:
                if output_name not in existing_outputs:
                    model_copy.graph.output.extend(
                        [onnx.ValueInfoProto(name=output_name)]
                    )
 
        session      = ort.InferenceSession(model_copy.SerializeToString())
        input_name   = session.get_inputs()[0].name
        output_names = [o.name for o in session.get_outputs()]
 
        # Final output name — the last declared output of the original model
        final_output_name = session.get_outputs()[0].name
 
        results = []
        for record in test_data:
            input_data  = np.expand_dims(record['input'].astype(np.float32), axis=0)
            raw_outputs = session.run(output_names, {input_name: input_data})
 
            # Build per-layer dict from all node outputs
            # ONNX exposes operations rather than layers, so each entry is one op
            per_layer = {}
            flat_parts = []
            for name, output in zip(output_names, raw_outputs):
                if output is None or not hasattr(output, '__len__'):
                    continue
                arr = np.array(output[0])
                per_layer[name] = arr.flatten().tolist()
                flat_parts.append(arr.flatten())
 
            flat = np.concatenate(flat_parts) if flat_parts else np.array([])
 
            final_idx   = output_names.index(final_output_name)
            final_output = raw_outputs[final_idx]
 
            results.append({
                'id':                record['id'],
                'input':             record['input'].tolist(),
                'label':             record.get('label'),
                'predicted':         int(np.argmax(final_output[0])),
                'correct':           int(np.argmax(final_output[0])) == record.get('label'),
                'activations':       flat,
                'layer_activations': per_layer,
            })
 
        return results

    def __run_pytorch_inference(self, test_data: list[dict]) -> list[dict]:
        # captured_activations is rebuilt per record inside the loop
        captured: dict[str, np.ndarray] = {}
 
        def make_hook(name: str):
            def hook(module, input, output):
                # output may be a tensor or a tuple (e.g. LSTM returns (out, hidden))
                if isinstance(output, tuple):
                    tensor = output[0]
                else:
                    tensor = output
                captured[name] = tensor.detach().numpy()[0]
            return hook
 
        hooks = []
        for name, module in self.model.named_modules():
            if name != '':
                hooks.append(module.register_forward_hook(make_hook(name)))
 
        results = []
        with torch.no_grad():
            for record in test_data:
                captured.clear()
 
                input_tensor = torch.tensor(
                    np.expand_dims(record['input'], axis=0),
                    dtype=torch.float32
                )
                output = self.model(input_tensor)
 
                # Build per-layer dict in hook-capture order
                per_layer = {
                    name: arr.flatten().tolist()
                    for name, arr in captured.items()
                }
 
                # Flat concatenation for distance analysis
                flat = np.concatenate([
                    arr.flatten() for arr in captured.values()
                ]) if captured else np.array([])
 
                predicted = int(torch.argmax(output[0]).item())
 
                results.append({
                    'id':                record['id'],
                    'input':             record['input'].tolist(),
                    'label':             record.get('label'),
                    'predicted':         predicted,
                    'correct':           predicted == record.get('label'),
                    'activations':       flat,
                    'layer_activations': per_layer,
                })
 
        # Always remove hooks after inference to prevent accumulation
        for hook in hooks:
            hook.remove()
 
        return results
    
    

    def load_dataset(self, file_path: str, label_column: Optional[str] = None) -> list[dict]:
        """
        Loads a dataset from a .csv or .npz file and returns a standardized
        list of records ready for inference.

        For CSV:
            label_column: name of the column containing the label e.g. 'label'
            all other columns are treated as input features

        For NPZ:
            expects 'x_test' and optionally 'y_test' arrays
            e.g. np.savez('data.npz', x_test=x_test, y_test=y_test)
        """
        ext = file_path.split('.')[-1].lower()
        if ext not in self.SUPPORTED_DATASET_FORMATS:
            raise ValueError(
                f"Unsupported dataset format: .{ext}. "
                f"Supported formats: {self.SUPPORTED_DATASET_FORMATS}"
            )

        loaders = {
            'csv': self.__load_csv_dataset,
            'npz': self.__load_npz_dataset,
        }
        return loaders[ext](file_path, label_column)

    def __load_csv_dataset(self, file_path: str, label_column: Optional[str]) -> list[dict]:
        df = pd.read_csv(file_path)

        # Validate label column if provided
        if label_column and label_column not in df.columns:
            raise ValueError(
                f"Label column '{label_column}' not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )

        records = []
        for idx, row in df.iterrows():
            if label_column:
                label = row[label_column]
                input_data = row.drop(label_column).to_numpy().astype(np.float32)
            else:
                label = None
                input_data = row.to_numpy().astype(np.float32)

            records.append({
                'id':    f"record_{idx}",
                'input': input_data,
                'label': int(label) if label is not None else None,
            })

        return records

    def __load_npz_dataset(self, file_path: str, label_column: Optional[str] = None) -> list[dict]:
        data = np.load(file_path, allow_pickle=False)

        # Validate expected keys
        if 'x_test' not in data:
            raise ValueError(
                f"NPZ file must contain 'x_test' array. "
                f"Found keys: {list(data.keys())}"
            )

        x_test = data['x_test']
        y_test = data['y_test'] if 'y_test' in data else None

        records = []
        for idx, input_data in enumerate(x_test):
            records.append({
                'id':    f"record_{idx}",
                'input': input_data.astype(np.float32),
                'label': int(y_test[idx]) if y_test is not None else None,
            })
        

        return records    #[:100]


    def run_full_inference( #TODO: why does this exist?
        self,
        dataset_path: str,
        label_column: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> dict:
        """
        Full pipeline: loads dataset, runs inference, returns results with summary.

        dataset_path: path to .csv or .npz file
        label_column: column name for labels in CSV files
        batch_size:   if provided, processes records in batches (useful for large datasets)

        Returns a dict with:
            - records:  list of records with activations and predictions
            - summary:  accuracy, total records, correct predictions etc.
        """
        records = self.load_dataset(dataset_path, label_column)

        if batch_size:
            results = self.__run_batched_inference(records, batch_size)
        else:
            results = self.run_inference(records)

        summary = self.__summarize_results(results)

        return {
            'inference_results': results,
            'summary': summary,
        }

    def __run_batched_inference(self, records: list[dict], batch_size: int) -> list[dict]: #TODO: why does this exist?
        """
        Splits records into batches and runs inference on each batch.
        Useful for large datasets that would be slow or memory intensive
        to run all at once.
        """
        results = []
        total = len(records)
        num_batches = (total + batch_size - 1) // batch_size  # ceiling division

        for batch_idx in range(num_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = records[start:end]

            print(f"  Processing batch {batch_idx + 1}/{num_batches} "
                  f"(records {start}-{end - 1})...")

            batch_results = self.run_inference(batch)
            results.extend(batch_results)

        return results

    def __summarize_results(self, results: list[dict]) -> dict:
        """
        Summarizes inference results including accuracy and per-class breakdown.
        Only calculates accuracy metrics if labels were provided in the dataset.
        """
        total = len(results)
        has_labels = all(r['label'] is not None for r in results)

        if not has_labels:
            return {
                'total_records':    total,
                'has_labels':       False,
            }

        correct = sum(1 for r in results if r['correct'])
        incorrect = total - correct
        accuracy = correct / total if total > 0 else 0.0

        # Per class breakdown
        class_results = {}
        for record in results:
            label = record['label']
            if label not in class_results:
                class_results[label] = {'total': 0, 'correct': 0}
            class_results[label]['total'] += 1
            if record['correct']:
                class_results[label]['correct'] += 1

        per_class_accuracy = {
            label: {
                'total':    stats['total'],
                'correct':  stats['correct'],
                'accuracy': stats['correct'] / stats['total'],
            }
            for label, stats in sorted(class_results.items())
        }

        return {
            'total_records':      total,
            'has_labels':         True,
            'correct':            correct,
            'incorrect':          incorrect,
            'accuracy':           accuracy,
            'per_class_accuracy': per_class_accuracy,
        }