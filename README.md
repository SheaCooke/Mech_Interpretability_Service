Goal:
- user provides a pytorch file that contains the state of a NN
- the program should read in that file, and create a custom representation, using the additions for observability
- the program should then read-in a test dataset and run the custom model through that
- during inference, the model should report information about the activations and path through the NN. this should be stored and associated with the record that was passed for inference
- After all the training records have been processed, some chart should be produced that shows what records where the most similar to eachother. this should be based on activations and the path through the NN.


Use cases:
- classification models: understand how resilient the model is to making a mistake between 2 classes? 
- RL models: 
- regression models: 



inference:
(linear transformation)
- input vector is multiplied by the weight matrix (columns = num of neurons in that layer. the rows = the num of expected input features)
- terms in the bias vector are added (1 value for each neuron)
- produces a vector of pre-activation values
(nonlinear transformation)
- the output (z) is then passed through an activation function (run once per layer. done with a vector for optimization, conceptually each neuron runs it. Activation func is applied elementwise (aka, iterates over all values within the vector))


Where do you emit a value to track the activation path through the network?

