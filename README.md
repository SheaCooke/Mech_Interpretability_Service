Goal:
- user provides a pytorch file that contains the state of a NN
- the program should read in that file, and create a custom representation, using the additions for observability
- the program should then read-in a test dataset and run the custom model through that
- during inference, the model should report information about the activations and path through the NN. this should be stored and associated with the record that was passed for inference
- After all the training records have been processed, some chart should be produced that shows what records where the most similar to eachother. this should be based on activations and the path through the NN.


Use cases:
- classification models: understand how resilient the model is to making a mistake between 2 classes?. verify model is grouping inputs logically. unexpected low distances can explain why misclassifications are happening. low distance between 2 records of different labels can indicate that the model needs more training. can help identify gaps in training if some records have an extremely high cosine distance. Identify circuits for different operations or categories.
- RL models: 
- regression models: 
- content moderation with LLMs?? for example, collect the vectors that represent activity for topics that the LLM is not supposed to discuss, then evaluate query responses for similarity to those collection of vectors before it is returned to the user


inference:
(linear transformation)
- input vector is multiplied by the weight matrix (columns = num of neurons in that layer. the rows = the num of expected input features)
- terms in the bias vector are added (1 value for each neuron)
- produces a vector of pre-activation values
(nonlinear transformation)
- the output (z) is then passed through an activation function (run once per layer. done with a vector for optimization, conceptually each neuron runs it. Activation func is applied elementwise (aka, iterates over all values within the vector))


similar vector = they triggered similar neurons at a similar magnitude 


is there a way to identify similarity|patterns at different layers?

are there any benefits to using both pre and post activation values in vector to determine similarity?




python3 -m venv venv
source venv/bin/activate
pip install --upgerade pip
pip install -r requirements.txt



-- On windows:
py --list
py -3.11 -m venv venv
source venv/Scripts/activate
python --version 
pip install -r requirements.txt


# Terminal 1 — backend
source venv/bin/activate
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Terminal 2 — frontend
npm install
npm run dev
npm run build



--------------
(currently only supports classification models, no user accounts or storing data for now)

model is loaded properly
- X run the training data through it
- X collect the activation vectors
- find some useful way to display that (in progress)
- Run diff components on Docker and K8s
- paginate the API between the backend and the frontend to match what the page size the user is viewing 
- Add documentation to UI (How-To page)
- code cleanup
-later layers likely capture higher level patterns, should experiment with weighting the layers differently when the values are converted to a vector. should be configurable through a parameter
- change title from NN analyzer to mech interpretability service / general UI cleanup
- add informational hover-over buttons on each widget, include what steps would be beneficial ("if this is not expected....", if clusters of diff labels are overlapping, then it is likely....)
- deploy on AWS
- security audit, including preventing AWS costs from going too high
- add demo pictures to github README (this file)
- logging
- unit tests
- are you sure? pop up after clicking reset

- Data visualizations

X -clustering of similarity, should be able to identify individual records in the graph (show record number and label) (color code clusters by label)
X -Comparisons between records that were incorrectly classified --> try to find some pattern that can provide guidance for training
X -can all be on the same page, just scroll down to see the different charts/graphs 
- add more detail to notifications: Found 1 similar pairs. --> add info about similarity metric used
- add "what to do with this info?" section

- layer-wise analysis (identify where patterns begin to emerge from raw data) and also cross-record comparison (current)


In mechanistic interpretability, this cross-record comparison is used for several specific purposes:
Probing and Concept Discovery: By comparing activations from records that share a common trait (e.g., all images of "stripes") against those that don't, researchers can identify Concept Activation Vectors (CAVs). These vectors represent the human-understandable concept within that layer's high-dimensional space.
Activation Distribution Analysis: Tools like NeuralDivergence use these distributions as a high-level summary to compare how different classes or instances (such as benign vs. adversarial images) are processed by the network.
Polysemanticity and SAEs: When training Sparse Autoencoders (SAEs), researchers analyze activations across thousands of different records to "disentangle" neurons that fire for multiple unrelated concepts.
Superposition Analysis: This involves studying how a model "packs" more concepts (features) into its activation space than it has neurons, which can only be observed by seeing how different records activate the same sets of neurons in different combinations.
Latent Clustering: Researchers use dimensionality reduction (like t-SNE or UMAP) on these collections of vectors to see if the model naturally clusters similar records, which indicates it has learned to generalize those categories. 


Planned Features:
- sliding bar on Similarity threshold should have 2 points so you can filter for an inclusive range
- MAX_DISPLAY should be replaced with max result per page. All pages contained within Similar Activation Pairs widget
- Add filtering option to Similar Activation Pairs widget
- caching per session to speed up switching between activation vector filtering options
- optimizations for running inference
- Cluster Plot hover featuer is broken
- cluster plot should have filter options for different labels
- user should be able to send something directly to the API from google colab (get all the activation vectors??)
- some way to identify common patterns
- diff pages for indeapth layer-wise analysis of individual records
- support for regression models 
- support updating parameters, or activation functions through the UI
- selecting a record number on the similarity pane should display the record
- similarity: more options than cosine distance (dot product)