import { useState } from "react";
import { Cpu } from "lucide-react";
import "./styles/global.css";
import {
  uploadModel, uploadDataset, runInference,
  fetchSimilarPairs, fetchClusterPlot, deleteSession,
  LayerDeviationData, IncorrectRecord, fetchLayerDeviation,
  fetchIncorrectRecords,
} from "./api/client";
import type {
  ModelData, InferenceSummary, SimilarPair,
  DatasetMeta, StatusMessage, Step, PredictionFilter, Page,
} from "./types";
import type { ClusterPlotData } from "./api/client";
import Header       from "./components/Header";
import Sidebar      from "./components/Sidebar";
import StatusBar    from "./components/StatusBar";
import ModelInfo    from "./components/ModelInfo";
import SummaryPanel from "./components/SummaryPanel";
import PairsPanel   from "./components/PairsPanel";
import ClusterPlot  from "./components/ClusterPlot";
import InstructionsPage from "./pages/InstructionsPage";
import InterpretingPage from "./pages/InterpretingPage";
import LayerDeviationPlot  from "./components/LayerDeviationPlot";

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>("home");

  const [sessionId,        setSessionId]        = useState<string | null>(null);
  const [modelData,        setModelData]        = useState<ModelData | null>(null);
  const [datasetMeta,      setDatasetMeta]      = useState<DatasetMeta | null>(null);
  const [summary,          setSummary]          = useState<InferenceSummary | null>(null);
  const [pairs,            setPairs]            = useState<SimilarPair[] | null>(null);
  const [clusterData,      setClusterData]      = useState<ClusterPlotData | null>(null);
  const [step,             setStep]             = useState<Step>("upload-model");
  const [loading,          setLoading]          = useState(false);
  const [status,           setStatus]           = useState<StatusMessage | null>(null);
  const [labelColumn,      setLabelColumn]      = useState("");
  const [thresholdLow,  setThresholdLow]  = useState(0.0);
  const [thresholdHigh, setThresholdHigh] = useState(0.2);
  const [predictionFilter, setPredictionFilter] = useState<PredictionFilter>("all");
  const [inferenceLimit, setInferenceLimit] = useState(0);

  const [incorrectRecords,  setIncorrectRecords]  = useState<IncorrectRecord[]>([]);
  const [deviationData,     setDeviationData]     = useState<LayerDeviationData | null>(null);
  const [deviationLoading,  setDeviationLoading]  = useState(false);

  function setErr(msg: string) { setStatus({ msg, type: "error"   }); setLoading(false); }
  function setOk (msg: string) { setStatus({ msg, type: "success" }); setLoading(false); }

  async function handleModelFile(file: File) {
    setLoading(true);
    setStatus({ msg: `Loading ${file.name}…`, type: "info" });
    try {
      const res = await uploadModel(file);
      setSessionId(res.session_id);
      setModelData(res.model_data);
      setStep("upload-dataset");
      setOk(`Model loaded — session ${res.session_id.slice(0, 8)}…`);
    } catch (e: any) { setErr(e.message); }
  }

  async function handleDatasetFile(file: File) {
    if (!sessionId) return;
    setLoading(true);
    setStatus({ msg: `Loading ${file.name}…`, type: "info" });
    try {
      const res = await uploadDataset(sessionId, file, labelColumn || undefined);
      setDatasetMeta({ filename: res.filename, num_records: res.num_records });
      setInferenceLimit(res.num_records); // default to all records
      setStep("inference");
      setOk(`Dataset loaded — ${res.num_records.toLocaleString()} records`);
    } catch (e: any) { setErr(e.message); }
  }

  async function handleRunInference() {
    if (!sessionId) return;
    setLoading(true);
    setSummary(null);
    setPairs(null);
    setIncorrectRecords([]);
    setClusterData(null);
    setStatus({ msg: "Running inference…", type: "info" });
    try {
      const res = await runInference(sessionId, inferenceLimit);
      setSummary(res.summary);
      setStep("analysis");

      // Immediately fetch incorrect records for layer-wise panel
      const inc = await fetchIncorrectRecords(sessionId);
      setIncorrectRecords(inc.records);


      setOk("Inference complete.");
    } catch (e: any) { setErr(e.message); }
  }

  async function handleFindPairs() {
    if (!sessionId) return;
    setLoading(true);
    setPairs(null);
    setStatus({ msg: `Computing similar pairs (${predictionFilter})…`, type: "info" });
    try {
      const res = await fetchSimilarPairs(sessionId, thresholdLow, thresholdHigh, predictionFilter);
      setPairs(res.pairs);
      setOk(`Found ${res.num_pairs} similar pairs (with filter for ${predictionFilter} inference results).`);
    } catch (e: any) { setErr(e.message); }
  }

  async function handleClusterPlot() {
    if (!sessionId) return;
    setLoading(true);
    setClusterData(null);
    setStatus({ msg: `Reducing dimensions (${predictionFilter})…`, type: "info" });
    try {
      const res = await fetchClusterPlot(sessionId, predictionFilter);
      setClusterData(res);
      setOk(`Cluster plot ready — ${res.points.length} points via ${res.method}.`);
    } catch (e: any) { setErr(e.message); }
  }


    async function handleRecordSelect(recordId: string) {
    if (!sessionId) return;
    setDeviationLoading(true);
    setDeviationData(null);
    try {
      const res = await fetchLayerDeviation(sessionId, recordId);
      setDeviationData(res);
    } catch (e: any) {
      setStatus({ msg: e.message, type: "error" });
    } finally {
      setDeviationLoading(false);
    }
  }

  async function handleReset() {
    if (sessionId) await deleteSession(sessionId).catch(() => {});
    setSessionId(null); setModelData(null); setDatasetMeta(null);
    setSummary(null); setPairs(null); setClusterData(null);
    setIncorrectRecords([]); setDeviationData(null);
    setStep("upload-model"); setStatus(null);
    setPredictionFilter("all");
  }
  const showAnalysis = step === "analysis";

  return (
    <div className="app">
      <Header
        step={step}
        sessionId={sessionId}
        currentPage={currentPage}
        onNavigate={setCurrentPage}
        onReset={handleReset}
      />
      {/* Instructions page */}
      {currentPage === "instructions" && <InstructionsPage />}
 
      {/* Interpreting results page */}
      {currentPage === "interpreting" && <InterpretingPage />}

      {/* Home page */}
      {currentPage === "home" && (
      <main className="main">
        {status && <StatusBar {...status} />}
        <div className="columns">
          <Sidebar
            step={step} loading={loading} sessionId={sessionId}
            datasetMeta={datasetMeta} labelColumn={labelColumn}
            inferenceLimit={inferenceLimit}
            thresholdLow={thresholdLow}
            thresholdHigh={thresholdHigh} 
            predictionFilter={predictionFilter}
            onLabelColumnChange={setLabelColumn}
            onInferenceLimitChange={setInferenceLimit}
            onThresholdChange={(low, high) => { setThresholdLow(low); setThresholdHigh(high); }}
            onPredictionFilterChange={setPredictionFilter}
            onModelFile={handleModelFile}
            onDatasetFile={handleDatasetFile}
            onRunInference={handleRunInference}
            onFindPairs={handleFindPairs}
            onClusterPlot={handleClusterPlot}
          />
          <div className="col-right">
              {/* Always-visible model info */}
              {modelData && <ModelInfo data={modelData} />}
 
              {/* Inference summary */}
              {summary && <SummaryPanel summary={summary} />}
 
              {/* ── General Analysis ── */}
              {showAnalysis && (
                <div className="analysis-section">
                  <h2 className="analysis-section-title">General Analysis</h2>
                  {pairs && <PairsPanel pairs={pairs} />}
                  {clusterData  && (
                    <ClusterPlot
                      points={clusterData.points}
                      method={clusterData.method}
                    />
                  )}
                </div>
              )}
 
              {/* ── Layer-wise Analysis ── */}
              {showAnalysis && (
                <div className="analysis-section">
                  <h2 className="analysis-section-title">Layer-Wise Analysis</h2>
                  {incorrectRecords.length === 0 ? (
                    <div className="analysis-section-empty">
                      No incorrectly classified records found
                    </div>
                  ) : (
                    <LayerDeviationPlot
                      sessionId={sessionId ?? ""}
                      incorrectRecords={incorrectRecords}
                      onRecordSelect={handleRecordSelect}
                      deviationData={deviationData}
                      loading={deviationLoading}
                    />
                  )}
            </div>
              )}
            {!modelData && (
                  <div className="empty-state">
                    <Cpu size={48} strokeWidth={1} />
                    <p>Upload a model to begin.</p>
                  </div>
            )}
          </div>
        </div>
      </main>
      )}
    </div>
  );
}