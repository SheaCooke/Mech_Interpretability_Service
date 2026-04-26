# NN Analyzer
 
A mechanistic interpretability tool for classification neural networks. NN Analyzer lets you load a trained model, run a labelled test dataset through it, and explore the internal representations the model builds — revealing *how* the model thinks, not just whether it gets the right answer.
 
---
 
## What is Mechanistic Interpretability?
 
Neural networks are often described as black boxes. You put an input in, a prediction comes out, and the internal process that produced that prediction is largely invisible. Accuracy metrics tell you *how often* the model is right, but they say almost nothing about *why* it is right, *why* it fails when it does, or whether it is making decisions for the right reasons.
 
Mechanistic interpretability is the discipline of opening that black box. It asks: what is the model actually computing, layer by layer, neuron by neuron? Which internal representations does it build, how does it group similar inputs, and at what point in the network does a wrong decision first take shape?
 
### Activations and Internal Representations
 
When an input is passed through a neural network, each layer transforms it and produces an output called an **activation** — a vector of numbers that encodes the model's representation of that input at that stage of processing. The early layers of a network tend to capture low-level features (edges, textures, basic patterns), while deeper layers capture increasingly abstract concepts relevant to the classification task.
 
Two inputs that produce similar activation vectors at a given layer are being processed similarly by the model at that point — the model has placed them in the same region of its internal representation space. This is a stronger and more informative relationship than two inputs simply sharing a predicted class. It means the model is using the same internal pathway to process both, which may or may not align with what we as humans consider them to have in common.
 
### The Geometry of Activation Space
 
The full set of activation vectors produced by a model across many inputs forms a high-dimensional geometry — a space where proximity means internal similarity. Mechanistic interpretability is largely the study of this geometry:
 
- **Tight, well-separated clusters** — the model has learned distinct, stable internal concepts for each class. It is not just memorising correct outputs; it is building geometrically separable representations.
- **Overlapping regions** — the model cannot internally distinguish two classes well, even if its accuracy appears acceptable. The overlap is a structural explanation for confusion between those classes.
- **Outlier points** — inputs the model processes very differently from all others in their class. These are often atypical examples, noisy inputs, or mislabelled records.
- **Incorrect predictions at boundaries** — errors that occur at the meeting point of two clusters indicate genuine ambiguity. The model is not randomly wrong; the input's features genuinely place it between two classes in the model's internal geometry.
- **Incorrect predictions deep inside a wrong cluster** — a more serious failure mode. The model has strongly committed to the wrong class internally, suggesting a feature or concept incorrectly associated across classes.
### Prototypes and Deviation
 
One of the most powerful tools in mechanistic interpretability is the concept of a **prototype** — the average internal representation of a class across many correctly classified examples. A prototype is what the model "expects" a given class to look like at each layer.
 
Comparing an individual record's layer-by-layer activations against the prototype for its true label reveals *where in the network* the model begins to process it incorrectly. If the deviation from the true-label prototype is low at the early layers but rises sharply at a specific hidden layer, that layer is where the model's internal representation of the input begins to diverge from the correct concept. This points directly to where an architectural change, regularisation strategy, or additional training data would be most effective.
 
### Why This Matters for Training
 
Accuracy alone is a blunt instrument for diagnosing model failures. A model with 95% accuracy may be getting the right answers for the wrong reasons — relying on spurious correlations in the training data that happen to generalise, or building fragile representations that will fail under distribution shift. Mechanistic interpretability provides a richer signal:
 
- **Systematic vs random errors** — if incorrect predictions cluster tightly together in activation space, the errors are not random noise. There is a specific pattern of inputs the model fails on, and understanding that pattern is actionable.
- **Layer attribution** — knowing which layer first produces a wrong representation tells you where in the architecture to intervene. Adding capacity, changing activation functions, or applying targeted regularisation at the responsible layer is a more principled approach than adjusting the full network.
- **Representation quality** — a model with well-separated class clusters in its intermediate layers is likely to generalise better than one that only separates classes at the final output. Monitoring cluster quality during training can give earlier feedback than validation accuracy alone.
- **Data quality signals** — inputs that appear far from their class cluster in activation space are candidates for mislabelled or ambiguous training examples. Cleaning or augmenting those examples directly addresses a root cause of model error.
- **Class confusion diagnosis** — when two classes consistently overlap in activation space, it may indicate that the training data does not contain enough distinguishing features, or that the model architecture does not have sufficient capacity to represent the difference. Either finding has a concrete remediation.
---
 
## What NN Analyzer Provides
 
NN Analyzer implements the core tools of mechanistic interpretability for classification models and exposes them through an interactive interface. The full analysis pipeline runs in four steps: upload a model, upload a test dataset, run inference, then explore the activation space with the analysis tools.
 
### General Analysis
 
**Cluster Plot**
 
All activation vectors are reduced to two dimensions using UMAP (or t-SNE as a fallback) and rendered as an interactive scatter plot. Each point is one record, coloured by its ground-truth label. Dim points with a red ring are incorrect predictions.
 
This is the starting point for any mechanistic analysis. A well-trained model will produce clearly separated colour regions. Overlap between colours indicates classes the model conflates internally. Incorrect predictions at the boundary between clusters are boundary-ambiguous cases; incorrect predictions inside a different class's cluster indicate a more serious representational failure.
 
The **prediction filter** — All / Correct only / Incorrect only — applies before the dimensionality reduction is computed. Generating the plot with "Incorrect only" shows the geometry of failures in isolation. If those incorrect predictions form their own tight cluster, the errors are structured and systematic, not random, which is one of the most diagnostically useful findings this tool can surface.
 
**Similar Pairs**
 
Every pair of records whose full-network cosine distance falls below a chosen threshold is listed. Cosine distance measures the angle between two activation vectors, making the comparison invariant to the overall scale of activations and sensitive only to the pattern of which neurons fire together.
 
Pairs where the two records carry different ground-truth labels are highlighted. These cross-label similar pairs are the mechanistic signature of class confusion: two inputs from different classes are being processed through the same internal pathway. This can indicate shared low-level features that the model has not learned to distinguish, or a gap in the training data that has left the decision boundary poorly defined in that region.
 
The threshold slider controls the sensitivity of the search. Starting at a higher threshold reveals the broad neighbourhood structure; lowering it progressively isolates the most tightly coupled pairs.
 
### Layer-Wise Analysis
 
**Prototype Deviation Chart**
 
After selecting any incorrectly classified record from the searchable list, the chart displays two lines across the model's layers:
 
- **True-label deviation** — the cosine distance between the selected record's activation at each layer and the prototype for its correct class. The prototype is the mean activation vector of all correctly classified records for that label.
- **Predicted-label deviation** — the cosine distance between the selected record's activation at each layer and the prototype for the class the model incorrectly predicted.
Reading the chart reveals the layer-by-layer story of a misclassification. A true-label deviation that starts low and rises sharply at a specific layer identifies that layer as the point where the model's internal representation of the input begins to diverge from the correct concept. If the predicted-label deviation converges toward zero at the same layer, that layer is where the model begins committing to the wrong class.
 
This is directly actionable for training. If most misclassifications show divergence at the same layer, that layer is the primary candidate for architectural improvement. If divergence appears at the earliest layers, the model may be lacking the capacity to represent fine-grained features at that depth, or the input preprocessing may be discarding discriminative information. If divergence only appears at the final layer, the earlier layers are building reasonable representations but the classification boundary itself is poorly positioned — additional training examples near the decision boundary may be sufficient to correct the errors.

## Setup
- py --list
- py -3.11 -m venv venv
- source venv/bin/activate
- pip install --upgerade pip
- pip install -r requirements.txt

### Terminal 1 — backend
- source venv/bin/activate
- uvicorn main:app --reload --host 0.0.0.0 --port 8000

### Terminal 2 — frontend
- cd frontend
- npm install
- npm run dev
- npm run build


## Planned Features:
- Make this project usable by an agent: expose API for uploading/inference/analysis then return the results in a way that is usable by an LLM, not just through the UI.
- Cluster Plot hover is broken - dosen't display record ids
- Provide option to weight outputs from different layers when performing analysis
- sliding bar on Similarity threshold should have 2 points so you can filter for an inclusive range
- add sliding bar to Run Inference pane to allow user to only run a subset of the data
- MAX_DISPLAY should be replaced with max result per page. All pages contained within Similar Activation Pairs widget
- Add filtering option to Similar Activation Pairs widget
- caching per session to speed up switching between activation vector filtering options
- optimizations for running inference
- cluster plot should have filter options for different labels
- Make the API callable from Google Colab to facilitate use while training NNs
- support for regression models 
- support updating parameters, or activation functions through the UI
- selecting a record number on the similarity pane should display the record
- similarity: more options than cosine distance (dot product)

