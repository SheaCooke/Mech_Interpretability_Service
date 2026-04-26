import { GitBranch, Hexagon, Filter, BarChart3, AlertTriangle, CheckCircle, HelpCircle } from "lucide-react";

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
  const icon = type === "positive"
    ? <CheckCircle size={13} />
    : type === "warning"
    ? <AlertTriangle size={13} />
    : <HelpCircle size={13} />;

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

export default function InterpretingPage() {
  return (
    <div className="page-content">

      <div className="page-hero">
        <h1 className="page-title">Interpreting Results</h1>
        <p className="page-subtitle">
          This page explains what each widget reveals about your model's internal
          representations and how to draw mechanistic insights from the outputs.
          All analysis in NN Analyzer operates on <strong>activation
          vectors</strong> — the full concatenation of every layer's output for
          a given input record.
        </p>
      </div>

      {/* What is an activation vector */}
      <section className="doc-section">
        <h2 className="doc-h2">What is an Activation Vector?</h2>
        <p className="doc-p">
          When a record is passed through the model, each layer transforms the
          data and produces an output — a list of numbers called an activation.
          NN Analyzer concatenates the activations from <em>all</em> layers into
          a single flat vector. This full-network activation vector is a
          high-dimensional fingerprint of how the model internally processes
          that specific input at every stage of computation.
        </p>
        <p className="doc-p">
          Two records with similar activation vectors are processed similarly by
          the model at every layer — not just at the output. This is a stronger
          claim than two records having the same predicted class. It means the
          model uses the same internal pathway to arrive at its decision.
        </p>
        <div className="callout-box">
          All similarity measurements use <strong>cosine distance</strong>, which
          measures the angle between two vectors rather than their absolute
          magnitude. This means the comparison is invariant to the overall scale
          of activations and focuses purely on the pattern of which neurons fire
          together.
        </div>
      </section>

      {/* Cluster plot */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <Hexagon size={16} style={{ display: "inline", marginRight: 8 }} />
          Cluster Plot
        </h2>
        <p className="doc-p">
          The cluster plot reduces the high-dimensional activation vectors to 2D
          using UMAP (or t-SNE if UMAP is unavailable) so they can be visualised
          as a scatter plot. Each point is one record. Points are coloured by
          their ground-truth label. Dim points with a red ring were predicted
          incorrectly.
        </p>

        <div className="insight-grid">
          <InsightCard icon={<CheckCircle size={16} />} title="Well-separated clusters">
            <p className="doc-p">
              When points of the same colour cluster tightly together and
              different colours are clearly separated, the model has learned
              strong, distinct internal representations for each class. This is a
              sign of good generalisation — the model is not just memorising
              outputs, it is building geometrically separable internal concepts.
            </p>
          </InsightCard>

          <InsightCard icon={<AlertTriangle size={16} />} title="Overlapping clusters">
            <p className="doc-p">
              When two or more label colours overlap significantly in the plot,
              the model cannot clearly distinguish those classes internally, even
              if its output accuracy appears reasonable. This overlap is a
              mechanistic explanation for why the model confuses those classes —
              their internal representations are too similar for the final layer
              to separate reliably.
            </p>
          </InsightCard>

          <InsightCard icon={<HelpCircle size={16} />} title="Incorrect predictions at cluster edges">
            <p className="doc-p">
              Incorrect predictions (dim, red-ringed points) that appear at the
              boundary between two clusters indicate ambiguous inputs — records
              the model internally represents as being between two classes. This
              is a meaningful finding: the model is not randomly wrong on these
              records, it is wrong because their features genuinely place them
              near a decision boundary.
            </p>
          </InsightCard>

          <InsightCard icon={<AlertTriangle size={16} />} title="Incorrect predictions inside a wrong cluster">
            <p className="doc-p">
              If an incorrect prediction appears deep inside a cluster of a
              different label colour, the model has strongly committed to the
              wrong class internally. This suggests a feature or concept the
              model has incorrectly associated across classes — a more serious
              failure mode than boundary ambiguity, and a target for deeper
              investigation.
            </p>
          </InsightCard>

          <InsightCard icon={<CheckCircle size={16} />} title="Sub-clusters within a class">
            <p className="doc-p">
              If a single label's points form multiple distinct sub-clusters, the
              model has identified meaningful sub-categories within that class
              even though they share a label. For example, in a digit classifier
              the model may represent differently-styled "4"s as distinct groups.
              This is not a failure — it shows the model has learned richer
              structure than the labels alone imply.
            </p>
          </InsightCard>

          <InsightCard icon={<HelpCircle size={16} />} title="Isolated outlier points">
            <p className="doc-p">
              Points that appear far from any cluster are records the model
              processes very differently from all others in their class. These
              are often unusual or atypical examples — noisy inputs, edge cases,
              or mislabelled records in the dataset. Hovering over them to get
              their record ID and then inspecting the raw input is a productive
              debugging step.
            </p>
          </InsightCard>
        </div>

        <div className="callout-box">
          <strong>Using the prediction filter with the cluster plot:</strong>{" "}
          Generate the plot with <em>All predictions</em> first to see the full
          picture. Then switch to <em>Incorrect only</em> and regenerate — the
          resulting plot shows only where the model fails and how those failures
          cluster. If incorrect predictions form their own tight cluster, there
          may be a systematic pattern to the errors rather than random noise.
        </div>
      </section>

      {/* Similar pairs */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <GitBranch size={16} style={{ display: "inline", marginRight: 8 }} />
          Similar Activation Pairs
        </h2>
        <p className="doc-p">
          The similar pairs table lists every pair of records whose cosine
          distance falls below the chosen threshold. A lower threshold means
          more similar — a distance of 0.0 would mean the two records are
          processed identically by the model at every layer.
        </p>

        <h3 className="doc-h3">Reading the threshold</h3>
        <p className="doc-p">
          There is no universally correct threshold — it depends on the model
          and dataset. A useful approach is to start high (0.3–0.4) to get a
          broad picture of which records are in the same neighbourhood, then
          lower it progressively to isolate the most tightly coupled pairs.
          Distances below 0.05 generally indicate that two records are
          activating almost the same internal pathway.
        </p>

        <div className="findings-list">
          <FindingRow
            type="neutral"
            label="Same label, low distance"
            description="The model processes these records through the same internal pathway. This is expected and confirms the model has a stable representation for this class."
          />
          <FindingRow
            type="warning"
            label="Different labels, low distance (orange rows)"
            description="The model processes inputs from two different classes almost identically internally, even though they have different ground-truth labels. This is the most mechanistically significant finding — it means the model cannot distinguish these classes by their internal representation, only by whatever small difference reaches the final layer."
          />
          <FindingRow
            type="positive"
            label="Many pairs for one label"
            description="A label that appears in many pairs at low threshold has a compact, consistent internal representation — the model has learned a tight concept for that class."
          />
          <FindingRow
            type="warning"
            label="Correct and incorrect predictions paired together"
            description="If a correct prediction and an incorrect prediction appear as a similar pair, the model is applying nearly the same internal computation to both records but producing different outputs. This can indicate sensitivity to very small input differences or instability in the final classification layer."
          />
        </div>

        <h3 className="doc-h3">Using the prediction filter with similar pairs</h3>
        <p className="doc-p">
          Switching the prediction filter to <em>Incorrect only</em> before
          running Find Similar Pairs is particularly revealing. If incorrect
          predictions cluster into similar pairs, the errors are not random —
          there is a specific internal representation the model produces for
          inputs it gets wrong. That representation can then be traced back
          through the layer-by-layer activations to identify which layer first
          produces the problematic pattern.
        </p>
      </section>

      {/* Inference summary */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <BarChart3 size={16} style={{ display: "inline", marginRight: 8 }} />
          Inference Summary
        </h2>
        <p className="doc-p">
          The inference summary provides per-class accuracy bars alongside
          overall accuracy. In the context of mechanistic interpretability,
          per-class accuracy differences are a starting point for investigation
          rather than an end result.
        </p>
        <div className="findings-list">
          <FindingRow
            type="neutral"
            label="Low accuracy for one class"
            description="Generate the cluster plot filtered to that class's incorrect predictions to see where in activation space the failures occur. If they cluster near another class, the model's internal representation for the two classes overlaps."
          />
          <FindingRow
            type="warning"
            label="High overall accuracy but poor per-class accuracy"
            description="The model may be over-representing the majority class internally. Check the cluster plot — if a minority class appears fragmented or embedded in another class's cluster, the model hasn't learned a distinct representation for it."
          />
          <FindingRow
            type="positive"
            label="Uniform per-class accuracy"
            description="The model has learned similarly strong representations for all classes. The cluster plot should show roughly equally-sized, well-separated clusters."
          />
        </div>
      </section>

      {/* Filter */}
      <section className="doc-section">
        <h2 className="doc-h2">
          <Filter size={16} style={{ display: "inline", marginRight: 8 }} />
          Prediction Filter
        </h2>
        <p className="doc-p">
          The prediction filter controls which activation vectors are included
          in both the similar pairs analysis and the cluster plot. It applies
          before any computation — a filtered cluster plot only reduces
          and displays vectors from the selected subset.
        </p>
        <div className="findings-list">
          <FindingRow
            type="neutral"
            label="All predictions"
            description="Use this first to get the full picture of the model's activation space. All records are included."
          />
          <FindingRow
            type="positive"
            label="Correct only"
            description="Shows the activation space of records the model gets right. Well-separated clusters here indicate the model has learned genuine class structure. Comparing this to the full plot reveals how much the incorrect predictions distort the overall geometry."
          />
          <FindingRow
            type="warning"
            label="Incorrect only"
            description="The most diagnostically useful filter. Shows only records the model predicted incorrectly. Tight clusters in this view indicate systematic failure modes — the model is wrong in a structured, consistent way rather than at random. Overlaps between label colours here directly identify which class pairs the model confuses."
          />
        </div>
      </section>

    </div>
  );
}