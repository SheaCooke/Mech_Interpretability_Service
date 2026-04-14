Goal:
- user provides a pytorch file that contains the state of a NN
- the program should read in that file, and create a custom representation, using the additions for observability
- the program should then read-in a test dataset and run the custom model through that
- during inference, the model should report information about the activations and path through the NN. this should be stored and associated with the record that was passed for inference
- After all the training records have been processed, some chart should be produced that shows what records where the most similar to eachother. this should be based on activations and the path through the NN.


Use cases:
- classification models: understand how resilient the model is to making a mistake between 2 classes?. verify model is grouping inputs logically. unexpected low distances can explain why misclassifications are happening. low distance between 2 records of different labels can indicate that the model needs more training. can help identify gaps in training if some records have an extremely high cosine distance
- RL models: 
- regression models: 
- content moderation with LLMs?? for example, collect the vectors that represent activity for topics that the LLM is not supposed to discuss, then evaluate query responses for similarity to those collection of vectors before it is returned to the user

mechanistic interpretability as a service

inference:
(linear transformation)
- input vector is multiplied by the weight matrix (columns = num of neurons in that layer. the rows = the num of expected input features)
- terms in the bias vector are added (1 value for each neuron)
- produces a vector of pre-activation values
(nonlinear transformation)
- the output (z) is then passed through an activation function (run once per layer. done with a vector for optimization, conceptually each neuron runs it. Activation func is applied elementwise (aka, iterates over all values within the vector))


Where do you emit a value to track the activation path through the network?


collect all activation values for each inference --> convert 2D list into a single vector --> cache the vector --> use cosine similarity to see how similar the activation path was to other records

similar vector = they triggered similar neurons at a similar magnitude 

TODO: later layers likely capture higher level patterns, should experiment with weighting the layers differently when the values are converted to a vector. should be configurable through a parameter

is there a way to identify similarity|patterns at different layers?

are there any benefits to using both pre and post activation values in vector to determine similarity?

TODO: remove code for training once the program is accepting pytorch files

analyzer functionality
- get activations
- create vectors
- store activations


TODO: API, K8s, front end

TODO: more organized way for tracking / downloading required packages


python3 -m venv venv
source venv/bin/activate
pip install --upgerade pip
pip install -r requirements.txt