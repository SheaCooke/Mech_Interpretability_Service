
## Description

This project aims to simplify the application of mechanistic interpretability principals for classification neural networks. As stated in the instruction page of the UI, the user first uploads a model file (currently only .keras models are supported) and a test dataset (.csv or .npz). When the records are passed through the model for inference, the output vectors of the activation function for each layer will be collected. These vectors can then be used for analysis to identify the layer at which a misclassification begins to diverge from the correct label. Having an understanding of this can guide the engineer in making targeted adjustments to improve the performance of the model.

https://cloudsecurityalliance.org/blog/2024/09/05/mechanistic-interpretability-101

## Setup for backend
from root of repo
- py -3.11 -m venv venv
- source venv/bin/activate
- pip install --upgerade pip
- pip install -r back_end/requirements.txt
- uvicorn back_end.api.main:app --reload --host 0.0.0.0 --port 8000

### frontend
- cd frontend
- npm install
- npm run dev


## Planned Features and Fixes:
- Support models from Pytorch and ONNX. Currently Keras is the only model lib supported
- Make this project usable by an agent: expose API for uploading/inference/analysis then return the results in a way that is usable by an LLM, not just through the UI.
- Provide option to weight outputs from different layers when performing analysis
- cluster plot should have filter options for different labels
- Make the API callable from Google Colab to facilitate use while training NNs
- support updating parameters, or activation functions through the UI
- selecting a record number on the similarity pane should display the record
- similarity: more options than cosine distance (dot product)
- support transformer based models
- support regression models
- Labels in Inference Summary should have label name, not just a number. Same with labels in layer-wise analysis section
- when searching Similar Activation Pairs, they should be sorted by relevance
- more efficient data structure for displaying/processing vectors
- dashboard for memory load / performance and progress with processing
- way to deal with excessive memory usage: write vectors to an Avro|Parquet file during inference. name should include the session_number.extension. The delete session method should delete these temp files. Intentionally cap RAM at some number. make this configurable. (lower RAM limit -> more use of files)
- update layer-wise analysis to either select individual records, or select entire set of misclassified labels and display all the layer-wise incorrect vectors (orange) along with the aggregate correct vector (blue)
- way to identify and visualize circuits within the network
- improve logging and testing




<img width="1572" height="857" alt="project_img_1" src="https://github.com/user-attachments/assets/edd98938-492d-4662-91fb-d3b6b1f6b562" />
----
<img width="1627" height="817" alt="project_img_2" src="https://github.com/user-attachments/assets/5decf290-07a2-4382-83dd-8ba681b0faad" />

----
<img width="1255" height="937" alt="Cluster_Plot" src="https://github.com/user-attachments/assets/c3c25214-fcc9-42c9-8f84-f7b5ec43ff29" />
----
<img width="1366" height="865" alt="layer-wise-analysis" src="https://github.com/user-attachments/assets/45bb8930-bd3f-47e4-81da-053384337094" />




