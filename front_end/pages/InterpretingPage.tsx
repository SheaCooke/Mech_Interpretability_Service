import { GitBranch, Hexagon, Filter, BarChart3, AlertTriangle, CheckCircle, HelpCircle, Layers, Wrench } from "lucide-react";

interface InsightCardProps {
  icon: React.ReactNode;
  title: string;
  children: React.ReactNode;
}

function InsightCard({ icon, title, children }: InsightCardProps) {
  return (
    <div className="insight-card">
      <div className="insight-card-header">
        <span className="insight-card-icon">{icon}</span>
        <h3 className="insight-card-title">{title}</h3>
      </div>
      <div className="insight-card-body">{children}</div>
    </div>
  );
}

interface FindingRowProps {
  type: "positive" | "warning" | "neutral";
  label: string;
  description: string;
}

function FindingRow({ type, label, description }: FindingRowProps) {
  const icon =
    type === "positive" ? <CheckCircle size={13} /> :
    type === "warning"  ? <AlertTriangle size={13} /> :
                          <HelpCircle size={13} />;

  return (
    <div className={`finding-row finding-${type}`}>
      <span className="finding-icon">{icon}</span>
      <div>
        <span className="finding-label">{label}: </span>
        <span className="finding-desc">{description}</span>
      </div>
    </div>
  );
}

interface TrainingActionProps {
  title: string;
  children: React.ReactNode;
}

function TrainingAction({ title, children }: TrainingActionProps) {
  return (
    <div className="training-action">
      <div className="training-action-header">
        <Wrench size={13} />
        <span className="training-action-title">{title}</span>
      </div>
      <div className="training-action-body">{children}</div>
    </div>
  );
}

export default function InterpretingPage() {
  return (
    <div className="page-content">

      <div className="page-hero">
        <h1 className="page-title">Interpreting Results</h1>
        <p className="page-subtitle">
          This page explains what each widget reveals about your model's internal
          representations, how to diagnose specific failure modes, and what
          concrete changes to make to your training setup based on what you find.
          All analysis in this program is based on <strong>activation
          vectors</strong> - the outputs of every layer for a given input record.
        </p>
      </div>

      <section className="doc-section">
        <h2 className="doc-h2">What is an Activation Vector?</h2>
        <p className="doc-p">
          When a record is passed through the model, each layer transforms the
          data and produces an output - a list of numbers called an activation.
          This program concatenates the activations from all layers into
          a single flat vector for general analysis, and also stores each layer's
          output separately for layer-wise analysis.
        </p>
        <p className="doc-p">
          Two records with similar full-network activation vectors are processed
          similarly by the model at every layer — not just at the output. This is
          a stronger claim than two records sharing the same predicted class. It
          means the model uses the same internal computational pathway, which may
          or may not align with what humans consider them to have in common.
        </p>
        <div className="callout-box">
          All similarity measurements use <strong>cosine distance</strong>, which
          measures the angle between two vectors rather than their absolute
          magnitude. This makes comparisons invariant to the overall scale of
          activations and sensitive only to the pattern of which neurons fire
          together.
        </div>
      </section>

      <section className="doc-section">
        <h2 className="doc-h2">
          <Hexagon size={16} style={{ display: "inline", marginRight: 8 }} />
          Cluster Plot
        </h2>
        <p className="doc-p">
          The cluster plot reduces all activation vectors to 2D using UMAP and renders them as an interactive
          scatter plot. Each point is one record, coloured by ground-truth label.
          Dim points with a red ring were predicted incorrectly.
        </p>

        <div className="insight-grid">
          <InsightCard icon={<CheckCircle size={16} />} title="Well-separated clusters">
            <p className="doc-p">
              Points of the same colour cluster tightly and different colours are
              clearly separated. The model has learned strong, distinct internal
              representations for each class — it is not just memorising outputs
              but building geometrically separable internal concepts.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                This is the target state. If accuracy is still not meeting
                requirements, the separation is good but the final classification
                boundary may be suboptimal. Consider label smoothing, a stronger
                output layer, or increased training epochs rather than
                architectural changes.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<AlertTriangle size={16} />} title="Overlapping clusters">
            <p className="doc-p">
              Two or more label colours mix significantly. The model cannot
              internally distinguish those classes, even if output accuracy
              appears reasonable. The overlap is the mechanistic explanation for
              confusion between those classes — their internal representations
              are too similar for the final layer to reliably separate.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                First check the data: do the overlapping classes share genuine
                visual or statistical similarity? If so, the model may need more
                capacity at the layers where the overlap originates — use the
                Layer-Wise Analysis to find that layer, then add neurons or depth
                there. If the data is clearly distinguishable, the model may be
                under-trained; increase epochs or learning rate warmup. Adding
                class-specific augmentation for the confused pair can also push
                the representations apart.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<HelpCircle size={16} />} title="Incorrect predictions at cluster edges">
            <p className="doc-p">
              Incorrect predictions (dim, red-ringed points) appear at boundaries
              between two clusters. The model is not randomly wrong — these inputs
              genuinely land between two classes in activation space, making them
              inherently ambiguous.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                Hover over boundary errors to get their record IDs, then inspect
                the raw inputs. If they are genuinely ambiguous examples, adding
                more training samples near the same decision boundary — or
                applying mixup augmentation between the two confused classes —
                can sharpen the boundary. If the inputs are clearly one class,
                the decision boundary is poorly positioned and the model likely
                needs more training data for the underrepresented class.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<AlertTriangle size={16} />} title="Incorrect predictions inside a wrong cluster">
            <p className="doc-p">
              An incorrect prediction appears deep inside a cluster of a different
              label colour. The model has strongly committed to the wrong class
              internally — a more serious failure than boundary ambiguity. This
              suggests a feature or concept incorrectly associated across classes.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                Use Layer-Wise Analysis on these records immediately. Deep
                misplacement usually means a specific early or mid-network layer
                is building the wrong representation. The deviation chart will
                show at which layer the record's activations diverge from the
                correct class prototype and converge toward the wrong one.
                Targeted regularisation (dropout, weight decay) at that layer,
                or adding a batch normalisation layer before it, can disrupt the
                incorrect feature association.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<CheckCircle size={16} />} title="Sub-clusters within a class">
            <p className="doc-p">
              A single label's points form multiple distinct sub-clusters. The
              model has identified meaningful sub-categories within that class
              even though they share a label. In a digit classifier, for example,
              differently-styled "4"s may form separate groups. This is not a
              failure — it shows the model has learned richer structure than the
              labels alone imply.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                Sub-clusters are generally healthy. If they are causing accuracy
                problems (misclassifications between sub-clusters of different
                classes), consider whether sub-labels would improve the training
                signal, or whether data augmentation within each sub-cluster
                would help the model generalise across the sub-types it has
                identified.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<HelpCircle size={16} />} title="Isolated outlier points">
            <p className="doc-p">
              Points appear far from any cluster. These are records the model
              processes very differently from all others in their class — often
              noisy, atypical, or mislabelled inputs.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                Hover over outlier points to get their record IDs and inspect the
                raw inputs. Confirmed mislabelled records should be corrected or
                removed from training data — they create conflicting gradient
                signals that damage representation quality for the entire class.
                Genuinely atypical but correctly labelled inputs may benefit from
                targeted augmentation to generate more similar training examples
                around that region of input space.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<AlertTriangle size={16} />} title="Fragmented minority class">
            <p className="doc-p">
              One class's points are scattered across the plot in small disconnected
              fragments rather than forming a coherent cluster. The model has not
              learned a unified concept for this class and treats different
              instances of it as unrelated.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                This is the activation-space signature of class imbalance. The
                model has not seen enough examples of this class to build a stable
                internal representation. Apply oversampling (SMOTE or random
                oversampling), class-weighted loss, or targeted data collection
                for the underrepresented class. Increasing the weight of the
                minority class in the loss function will encourage the model to
                devote more representational capacity to distinguishing it.
              </p>
            </TrainingAction>
          </InsightCard>

          <InsightCard icon={<AlertTriangle size={16} />} title="All classes merged into one region">
            <p className="doc-p">
              The plot shows a single undifferentiated blob regardless of label
              colour. The model has not learned to distinguish classes at all in
              its internal representations — it is likely mapping most inputs to
              similar activations regardless of their content.
            </p>
            <TrainingAction title="What to do">
              <p className="doc-p">
                This usually indicates a training failure: vanishing gradients,
                an overly high learning rate that disrupts early training, or a
                model that has collapsed to predicting one class for everything.
                Check that the loss is decreasing during training, reduce the
                learning rate, add gradient clipping, and verify the data
                preprocessing is correct. A merged blob on a model that reports
                high accuracy is a red flag — the model may be exploiting a
                class imbalance rather than learning genuine features.
              </p>
            </TrainingAction>
          </InsightCard>
        </div>

        <div className="callout-box">
          <strong>Workflow recommendation:</strong> Generate the cluster plot
          three times — once with <em>All predictions</em>, once with{" "}
          <em>Correct only</em>, and once with <em>Incorrect only</em>. The
          comparison between the three views is more diagnostic than any single
          view alone. Well-separated correct predictions with tight incorrect
          clusters that overlap at specific class boundaries is the most
          actionable pattern: it tells you exactly which class pair needs
          attention and that the rest of the model is functioning well.
        </div>
      </section>

      {/* ── Similar Pairs ─────────────────────────────────────────────────── */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <GitBranch size={16} style={{ display: "inline", marginRight: 8 }} />
          Similar Activation Pairs
        </h2>
        <p className="doc-p">
          The similar pairs table lists every pair of records whose cosine
          distance falls below the chosen threshold. A lower threshold means
          more similar — a distance of 0.0 would mean the two records are
          processed identically by the model at every layer. This widget can be used as a 
          more detailed version of the cluster plot.
        </p>

        <h3 className="doc-h3">Reading the threshold</h3>
        <p className="doc-p">
          There is no universally correct threshold — it depends on the model
          and dataset. A useful approach is to start at 0.3–0.4 to get a broad
          picture, then lower progressively to isolate the most tightly coupled
          pairs. Distances below 0.05 generally indicate that two records are
          activating almost the same internal pathway.
        </p>

        <div className="findings-list">
          <FindingRow
            type="neutral"
            label="Same label, low distance"
            description="The model processes these records through the same internal pathway. This is expected and confirms the model has a stable, consistent representation for this class."
          />
          <FindingRow
            type="warning"
            label="Different labels, low distance (orange rows)"
            description="The model processes inputs from two different classes almost identically. This is the most mechanistically significant finding — it means the model cannot distinguish these classes by their internal representation. Use Layer-Wise Analysis on these records to identify which layer first fails to separate them."
          />
          <FindingRow
            type="positive"
            label="Many same-label pairs at low threshold"
            description="A label with many tightly coupled pairs has a compact, consistent internal representation — the model has learned a well-defined concept for this class."
          />
          <FindingRow
            type="warning"
            label="Correct and incorrect predictions paired together"
            description="The model applies nearly identical internal computation to both records but produces different outputs. This indicates instability at the final classification layer — small differences in input produce different predictions despite identical internal processing. Increasing model confidence through temperature scaling or label smoothing may help."
          />
        </div>
      </section>

      {/* ── Layer-Wise Analysis ───────────────────────────────────────────── */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <Layers size={16} style={{ display: "inline", marginRight: 8 }} />
          Layer-Wise Analysis — Prototype Deviation Chart
        </h2>
        <p className="doc-p">
          The prototype deviation chart is the most granular diagnostic tool in
          NN Analyzer. For a selected incorrectly classified record it shows two
          lines, one per layer:
        </p>
        <ul className="doc-list">
          <li>
            <CheckCircle size={12} />
            <span>
              <strong style={{ color: "#7c6af7" }}>True-label deviation</strong> —
              the cosine distance between the record's activation at each layer
              and the <em>prototype</em> for its correct class. The prototype is
              the mean activation vector of all correctly classified records for
              that label.
            </span>
          </li>
          <li>
            <AlertTriangle size={12} />
            <span>
              <strong style={{ color: "#f7836a" }}>Predicted-label deviation</strong> —
              the cosine distance between the record's activation at each layer
              and the prototype for the class the model incorrectly predicted.
            </span>
          </li>
        </ul>
        <p className="doc-p">
          The x-axis is the model's layers in order from input to output. The
          y-axis is cosine distance — a value of 0.0 means the record's
          activation at that layer is identical to the prototype; a higher value
          means it is more different.
        </p>

        <h3 className="doc-h3">Reading the chart — key patterns</h3>

        <div className="findings-list">

          <FindingRow
            type="warning"
            label="True-label deviation rises sharply at a specific layer"
            description="The record's activations were tracking the correct class up to that layer, then diverged. That layer is where the misclassification originates. This is the most precise finding the tool can produce — it narrows the cause of a specific error to a single layer of the network."
          />

          <FindingRow
            type="warning"
            label="Predicted-label deviation converges toward zero at the same layer"
            description="The record is simultaneously diverging from the correct class and converging toward the wrong class at the same layer. This double crossing is the clearest possible indicator of a representational failure at that specific layer — the layer is actively pulling the record toward the wrong class."
          />

          <FindingRow
            type="neutral"
            label="Both lines remain high throughout"
            description="The record does not strongly resemble either the correct or predicted class prototype at any layer. The model is uncertain throughout the network, not just at the output. This often indicates the input is genuinely unusual — consider inspecting it for noise or mislabelling."
          />

          <FindingRow
            type="neutral"
            label="True-label deviation stays low until the final layer"
            description="The intermediate layers are building a reasonable representation of the correct class, but the final classification layer misassigns it. The earlier layers are healthy — the problem is in the decision boundary itself, not the feature extraction."
          />

          <FindingRow
            type="warning"
            label="True-label deviation is high from the very first layer"
            description="The model fails to build a correct-class representation from the beginning. This either means the input preprocessing is discarding discriminative information, the earliest layers lack sufficient capacity to represent the features that distinguish this class, or this input is simply very different from all training examples of its class."
          />

          <FindingRow
            type="positive"
            label="True-label deviation decreases at deeper layers"
            description="The model is partially recovering toward the correct class as processing deepens. The later layers are compensating for an earlier incorrect representation. This is a sign of model resilience, but the early divergence is still worth addressing."
          />

        </div>

        <h3 className="doc-h3">Translating findings into training changes</h3>

        <div className="training-actions-grid">

          <TrainingAction title="Divergence at an early layer (input → first hidden)">
            <p className="doc-p">
              The model is failing to extract the features needed to distinguish
              this class from the first transformation. Consider: adding more
              neurons to the first hidden layer to increase its representational
              capacity; changing the activation function in early layers from
              ReLU to GELU or ELU to allow richer gradient flow; reviewing input
              normalisation to ensure discriminative features are not being
              compressed out of range; or adding a batch normalisation layer
              after the first hidden layer to stabilise early activations.
            </p>
          </TrainingAction>

          <TrainingAction title="Divergence at a middle hidden layer">
            <p className="doc-p">
              The early feature extraction is working but an intermediate
              abstraction layer is losing the class signal. This is the most
              common failure mode and has the most targeted remediation. Add a
              skip connection (residual connection) around that layer so the
              earlier representation is preserved. Alternatively, increase the
              width of that specific layer rather than the whole network, add
              dropout before it to prevent over-reliance on a narrow set of
              neurons, or insert a batch normalisation layer to prevent the
              representation from drifting during training.
            </p>
          </TrainingAction>

          <TrainingAction title="Divergence only at the final layer">
            <p className="doc-p">
              The feature extraction pipeline is working correctly — the model
              builds a reasonable intermediate representation of the correct
              class, but the final classification boundary is in the wrong place.
              The most effective interventions are: adding more training examples
              near the confused class boundary; applying label smoothing to
              prevent the final layer from becoming overconfident on training
              examples; using a higher weight decay specifically on the final
              layer's weights; or training the final layer for additional epochs
              with the earlier layers frozen.
            </p>
          </TrainingAction>

          <TrainingAction title="Divergence at the same layer across many records">
            <p className="doc-p">
              If you select multiple incorrect records and observe the deviation
              consistently rising at the same layer, that layer is a systematic
              bottleneck for the model. This is the strongest possible signal for
              an architectural intervention at a specific location. Widening that
              layer, replacing it with a more expressive block, or adding a
              parallel pathway around it are all well-motivated changes. Document
              the layer name from the x-axis and make it the focus of your next
              training experiment.
            </p>
          </TrainingAction>

          <TrainingAction title="Predicted-label deviation is lower than true-label deviation at every layer">
            <p className="doc-p">
              The model represents this record as more similar to the wrong class
              than the correct class at every layer of the network — a deep,
              pervasive misrepresentation. This record's input features are
              genuinely closer to the wrong class in the model's learned
              representation. Review whether the training data for the two
              confused classes is sufficiently diverse, whether the two classes
              genuinely share features that require higher-level discrimination to
              separate, and whether adding a contrastive loss term would help push
              the representations of the two classes further apart.
            </p>
          </TrainingAction>

          <TrainingAction title="High deviation from both prototypes throughout">
            <p className="doc-p">
              The record does not resemble either class prototype at any layer.
              Before making architectural changes, investigate the input itself —
              this pattern is common with mislabelled training data, corrupted
              inputs, or out-of-distribution test examples. If the input appears
              correct, consider whether the prototype for its class is being
              dominated by a sub-type of that class the model has learned to
              represent strongly, leaving this variant without a good prototype
              match. Generating more training examples similar to this input is
              usually the most effective response.
            </p>
          </TrainingAction>

        </div>

        <div className="callout-box">
          <strong>Recommended workflow for layer-wise analysis:</strong> Start by
          selecting several incorrectly predicted records that share the same
          true label and observe whether their deviation profiles show a
          consistent pattern. If the same layer diverges across multiple records
          of the same class, that is a systematic finding about the class
          representation, not a one-off anomaly. Cross-reference with the cluster
          plot — if the incorrect records for that class appear as a distinct
          sub-cluster far from the main class cluster, the layer-wise analysis
          will almost always show early divergence, confirming an architectural
          bottleneck rather than a data quality issue.
        </div>
      </section>

      {/* ── Inference Summary ─────────────────────────────────────────────── */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <BarChart3 size={16} style={{ display: "inline", marginRight: 8 }} />
          Inference Summary
        </h2>
        <p className="doc-p">
          The inference summary provides per-class accuracy bars alongside
          overall accuracy. In the context of mechanistic interpretability,
          per-class accuracy differences are a starting point for investigation
          rather than an end result — the cluster plot and layer-wise analysis
          explain the mechanism behind the numbers.
        </p>
        <div className="findings-list">
          <FindingRow
            type="neutral"
            label="Low accuracy for one class"
            description="Generate the cluster plot filtered to that class's incorrect predictions. If they cluster near another class, the two representations overlap and the layer-wise analysis will identify which layer is responsible. If they are scattered, the model has not learned a coherent concept for this class at all — a data volume or diversity problem."
          />
          <FindingRow
            type="warning"
            label="High overall accuracy but poor per-class accuracy"
            description="The model may be over-representing the majority class. Check whether the minority class appears fragmented in the cluster plot. If so, class-weighted loss, oversampling, or targeted data collection for the underrepresented class are the right interventions — architectural changes alone will not fix a data imbalance."
          />
          <FindingRow
            type="positive"
            label="Uniform per-class accuracy"
            description="The model has learned similarly strong representations for all classes. The cluster plot should show roughly equally-sized, well-separated clusters. If accuracy is still not at the required level, the issue is uniform rather than class-specific — consider increasing model capacity globally or training for longer."
          />
        </div>
      </section>

      {/* ── Prediction Filter ─────────────────────────────────────────────── */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <Filter size={16} style={{ display: "inline", marginRight: 8 }} />
          Prediction Filter
        </h2>
        <p className="doc-p">
          The prediction filter controls which activation vectors are included in
          both the similar pairs analysis and the cluster plot. It applies before
          any computation — a filtered cluster plot only reduces and displays
          vectors from the selected subset.
        </p>
        <div className="findings-list">
          <FindingRow
            type="neutral"
            label="All predictions"
            description="Use this first to see the complete activation space. All records are included. This is the baseline view against which filtered views should be compared."
          />
          <FindingRow
            type="positive"
            label="Correct only"
            description="Shows the activation space of records the model gets right. Well-separated clusters here confirm the model has learned genuine class structure. The geometry of this view represents the model at its best — the target state for all inputs."
          />
          <FindingRow
            type="warning"
            label="Incorrect only"
            description="The most diagnostically useful filter. Tight clusters in this view indicate systematic failure modes. Overlaps between label colours identify exactly which class pairs the model confuses. The records in this view are the primary candidates for layer-wise analysis."
          />
        </div>
      </section>

    </div>
  );
}