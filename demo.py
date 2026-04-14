import numpy as np
from neural_network import NeuralNetwork
from analyzer import Analyzer
from model_processor.model_processor import Model_Processor


# def demo_xor():
#     print("=" * 60)
#     print("DEMO 1: XOR Problem")
#     print("=" * 60)
 
#     X = np.array([[0, 0],
#                   [0, 1],
#                   [1, 0],
#                   [1, 1]], dtype=float)

#     Y = np.array([[0], [1], [1], [0]], dtype=float)
 
#     model = NeuralNetwork(
#         layer_configs=[
#             {"input_size": 2,  "output_size": 8,  "activation": "tanh"},
#             {"input_size": 8,  "output_size": 4,  "activation": "tanh"},
#             {"input_size": 4,  "output_size": 1,  "activation": "sigmoid"},
#         ],
#         loss="binary_crossentropy",
#         optimizer="adam",
#         optimizer_params={"lr": 0.01},
#     )
 
#     model.train(X, Y, epochs=3000, verbose=True, print_every=500)

#     analyzer = Analyzer(model, X)
#     print(f'activations: {analyzer.post_activations}')
#     print(f'vector: {analyzer.post_activations_vector}')


#     # analyzer = Analyzer(model)

#     # activations = analyzer.get_post_activations(X)

#     # print('--activations start --')

#     # for name, value in activations.items():
#     #     print(f'{name}  {value}')

#     # print('--activations end --')
    
 
#     print("\nPredictions:")
#     for xi, yi in zip(X, Y):
#         pred = model.predict(xi.reshape(1, -1))[0, 0]
#         print(f"  Input: {xi.astype(int)} | Target: {int(yi[0])} | Predicted: {pred:.4f}")
 
#     model.save("/tmp/xor_model.pkl")
#     return model
 


def demo_xor():
    print("=" * 60)
    print("DEMO 1: XOR Problem (4 features, 16 records)")
    print("=" * 60)
 
    # All 16 unique combinations of 4 binary features
    X = np.array([[0, 0, 0, 0],
                  [0, 0, 0, 1],
                  [0, 0, 1, 0],
                  [0, 0, 1, 1],
                  [0, 1, 0, 0],
                  [0, 1, 0, 1],
                  [0, 1, 1, 0],
                  [0, 1, 1, 1],
                  [1, 0, 0, 0],
                  [1, 0, 0, 1],
                  [1, 0, 1, 0],
                  [1, 0, 1, 1],
                  [1, 1, 0, 0],
                  [1, 1, 0, 1],
                  [1, 1, 1, 0],
                  [1, 1, 1, 1]], dtype=float)
 
    # Label is 1 if the number of 1s across all features is odd, else 0
    # This is the natural generalisation of XOR to multiple inputs
    Y = (X.sum(axis=1) % 2).reshape(-1, 1)
 
    print("Dataset:")
    for xi, yi in zip(X, Y):
        print(f"  {xi.astype(int)} → {int(yi[0])}")
 
    model = NeuralNetwork(
        #output size = number of weights, input size = number of neurons
        #the output size of 1 layer must equal the input size in the next layer. 
        # outputs of the activation function = the number of weights
        #The input size in the first layer will be the number of features
        layer_configs=[
            {"input_size": 4,  "output_size": 8,  "activation": "tanh"},
            {"input_size": 8,  "output_size": 4,  "activation": "tanh"},
            {"input_size": 4,  "output_size": 1,  "activation": "sigmoid"},
        ],
        loss="binary_crossentropy",
        optimizer="adam",
        optimizer_params={"lr": 0.01},
    )
 
    model.train(X, Y, epochs=5000, verbose=True, print_every=500)

    analyzer = Analyzer(model, X)
    #print(f'activations: {analyzer.post_activations}')
    print(f'num vector: {len(analyzer.activation_vectors)}')
    for key, value in analyzer.activation_vectors.items():
        print(f'vector: {key}')
        print(f'first: {key[0]}')
        print(f'record: {value}')
        break

 
    print("\nPredictions:")
    for xi, yi in zip(X, Y):
        pred = model.predict(xi.reshape(1, -1))[0, 0]
        label = "✓" if round(pred) == int(yi[0]) else "✗"
        print(f"  {label} Input: {xi.astype(int)} | Target: {int(yi[0])} | Predicted: {pred:.4f}")
 
    model.save("/tmp/xor_model.pkl")
    return model


 
# =============================================================================
# Demo 2: Spiral Dataset (multi-class classification)
# =============================================================================
 
def make_spiral_data(n_points=100, n_classes=3):
    X = np.zeros((n_points * n_classes, 2))
    Y = np.zeros(n_points * n_classes, dtype=int)
    for c in range(n_classes):
        r = np.linspace(0.0, 1, n_points)
        t = np.linspace(c * 4, (c + 1) * 4, n_points) + np.random.randn(n_points) * 0.2
        X[c * n_points:(c + 1) * n_points] = np.column_stack([r * np.sin(t), r * np.cos(t)])
        Y[c * n_points:(c + 1) * n_points] = c
    return X, Y
 
 
def demo_spiral():
    print("\n" + "=" * 60)
    print("DEMO 2: Spiral Dataset (3-class classification)")
    print("=" * 60)
 
    np.random.seed(42)
    X, Y = make_spiral_data(n_points=100, n_classes=3)
 
    model = NeuralNetwork(
        layer_configs=[
            {"input_size": 2,   "output_size": 64,  "activation": "relu"},
            {"input_size": 64,  "output_size": 64,  "activation": "relu"},
            {"input_size": 64,  "output_size": 3,   "activation": "softmax"},
        ],
        loss="categorical_crossentropy",
        optimizer="adam",
        optimizer_params={"lr": 0.001},
    )
 
    model.train(X, Y, epochs=2000, batch_size=32, verbose=True, print_every=400)
 
    final_acc = model._accuracy(X, Y)
    print(f"\nFinal training accuracy: {final_acc:.4f}")
    return model
 
 
def demo_model_ingest():
    mp = Model_Processor()
    print('weights ----------------- ')
    print(len(mp.weights))
    print('biases ----------------- ')
    print(len(mp.biases))
    print('activation_functions ----------------- ')
    print(mp.activation_functions)
    print('parameters ----------------- ')
    print(mp.parameters)





 
if __name__ == "__main__":
    # np.random.seed(0)
    # demo_xor()
    #demo_spiral()
    demo_model_ingest()