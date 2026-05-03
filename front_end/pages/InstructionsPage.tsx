import { Upload, Database, Play, GitBranch, ArrowRight } from "lucide-react";

export default function InstructionsPage() {
  return (
    <div className="page-content">

      <div className="page-hero">
        <h1 className="page-title">Instructions</h1>
        <p className="page-subtitle">
          NN Analyzer is a mechanistic interpretability tool for classification
          models. It lets you load a trained model, run a labelled test dataset
          through it, and explore the internal activation vectors the model
          produces — revealing <em>how</em> the model represents and separates
          different classes, not just <em>whether</em> it gets the right answer.
        </p>
      </div>

      {/* What is mechanistic interpretability */}
      <section className="doc-section">
        <h2 className="doc-h2">What is Mechanistic Interpretability?</h2>
        <p className="doc-p">
          A neural network classifies inputs by transforming them through
          successive layers. Each layer produces an <strong>activation
          vector</strong> — a list of numbers encoding the model's internal
          representation of that input at that depth. Mechanistic
          interpretability is the study of those internal representations:
          which inputs activate the same neurons, how the model groups similar
          concepts, and where errors originate in the network's internal
          geometry.
        </p>
        <p className="doc-p">
          NN Analyzer makes this concrete. Instead of treating the model as a
          black box, you can inspect the geometry of its activation space,
          identify which records the model represents similarly regardless of
          their true label, and compare the representations of correct
          predictions against incorrect ones.
        </p>
      </section>

      {/* Step by step */}
      <section className="doc-section">
        <h2 className="doc-h2">Step-by-Step Guide</h2>

        <div className="step-cards">

          <div className="step-card">
            <div className="step-card-icon"><Upload size={18} /></div>
            <div className="step-card-body">
              <h3 className="step-card-title">
                <span className="step-card-num">01</span> Upload a Model
              </h3>
              <p className="step-card-desc">
                Drag and drop or click to upload a trained classification model.
                Supported formats are <code>.keras</code>, <code>.onnx</code>,{" "}
                <code>.pt</code>, and <code>.pth</code>. The model architecture
                is displayed on the right — verify the layer structure and
                parameter count before proceeding.
              </p>
              <div className="tip-box">
                <strong>Tip:</strong> Models saved from Google Colab are
                supported. Download the <code>.keras</code> file and upload it
                directly.
              </div>
            </div>
          </div>

          <div className="step-card">
            <div className="step-card-icon"><Database size={18} /></div>
            <div className="step-card-body">
              <h3 className="step-card-title">
                <span className="step-card-num">02</span> Upload a Dataset
              </h3>
              <p className="step-card-desc">
                Upload a test dataset in <code>.csv</code> or <code>.npz</code>{" "}
                format. For CSV files, enter the name of the label column before
                uploading — this enables accuracy metrics and the prediction
                filter. For NPZ files the tool expects arrays named{" "}
                <code>x_test</code> and optionally <code>y_test</code>.
              </p>
              <div className="tip-box">
                <strong>Tip:</strong> Always use a held-out test set, not
                training data. Activation patterns on training data may not
                reflect how the model generalises.
              </div>
            </div>
          </div>

          <div className="step-card">
            <div className="step-card-icon"><Play size={18} /></div>
            <div className="step-card-body">
              <h3 className="step-card-title">
                <span className="step-card-num">03</span> Run Inference
              </h3>
              <p className="step-card-desc">
                Click <strong>Run Inference</strong> to pass every record
                through the model. The tool captures the full activation vector
                at every layer for every record. An accuracy summary is shown,
                broken down per class if labels were provided.
              </p>
              <div className="tip-box">
                <strong>Tip:</strong> Activation vectors are held in memory for
                the session — no data is written to disk. Refreshing the page
                clears all results.
              </div>
            </div>
          </div>

          <div className="step-card">
            <div className="step-card-icon"><GitBranch size={18} /></div>
            <div className="step-card-body">
              <h3 className="step-card-title">
                <span className="step-card-num">04</span> Analyse Activations
              </h3>
              <p className="step-card-desc">
                Two analysis tools are available. Use the{" "}
                <strong>Prediction Filter</strong> dropdown to restrict analysis
                to correct predictions, incorrect predictions, or all records
                before running either tool.
              </p>
              <ul className="doc-list">
                <li>
                  <ArrowRight size={12} />
                  <span>
                    <strong>Find Similar Pairs</strong> — set a cosine distance
                    threshold and find every pair of records whose full
                    activation vectors are closer than that threshold. Pairs
                    where the two records have different labels are highlighted
                    in orange — these are the most interpretability-relevant
                    results.
                  </span>
                </li>
                <li>
                  <ArrowRight size={12} />
                  <span>
                    <strong>Generate Cluster Plot</strong> — reduces all
                    activation vectors to 2D using UMAP (or t-SNE as a
                    fallback) and renders an interactive scatter plot. Points
                    are coloured by label. Hover over any point to see its
                    record ID, label, and whether the prediction was correct.
                    Dim points with a red ring are incorrect predictions.
                  </span>
                </li>
              </ul>
            </div>
          </div>

        </div>
      </section>

      {/* Supported formats */}
      <section className="doc-section">
        <h2 className="doc-h2">Supported File Formats</h2>
        <div className="format-grid">
          {[
            { ext: ".keras",     desc: "Keras 3.x native format. Recommended for models trained with TensorFlow / Keras." },
            { ext: ".csv",       desc: "Tabular dataset. All columns except the label column are treated as input features." },
            { ext: ".npz",       desc: "NumPy compressed archive. Must contain x_test and optionally y_test arrays." },
          ].map(f => (
            <div key={f.ext} className="format-row">
              <code className="format-ext">{f.ext}</code>
              <span className="format-desc">{f.desc}</span>
            </div>
          ))}
        </div>
      </section>

    </div>
  );
}