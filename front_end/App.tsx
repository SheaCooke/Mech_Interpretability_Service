import { useState } from "react";
import { Cpu } from "lucide-react";
import "./styles/global.css";
import {
  uploadModel, uploadDataset, runInference,
  fetchSimilarPairs, fetchClusterPlot, deleteSession,
} from "./api/client";
import type {
  ModelData, InferenceSummary, SimilarPair,
  DatasetMeta, StatusMessage, Step, PredictionFilter,
} from "./types";
import type { ClusterPlotData } from "./api/client";
import Header       from "./components/Header";
import Sidebar      from "./components/Sidebar";
import StatusBar    from "./components/StatusBar";
import ModelInfo    from "./components/ModelInfo";
import SummaryPanel from "./components/SummaryPanel";
import PairsPanel   from "./components/PairsPanel";
import ClusterPlot  from "./components/ClusterPlot";

export default function App() {
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
  const [threshold,        setThreshold]        = useState(0.1);
  const [predictionFilter, setPredictionFilter] = useState<PredictionFilter>("all");

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
      setStep("inference");
      setOk(`Dataset loaded — ${res.num_records.toLocaleString()} records`);
    } catch (e: any) { setErr(e.message); }
  }

  async function handleRunInference() {
    if (!sessionId) return;
    setLoading(true);
    setSummary(null);
    setPairs(null);
    setClusterData(null);
    setStatus({ msg: "Running inference…", type: "info" });
    try {
      const res = await runInference(sessionId);
      setSummary(res.summary);
      setStep("analysis");
      setOk("Inference complete.");
    } catch (e: any) { setErr(e.message); }
  }

  async function handleFindPairs() {
    if (!sessionId) return;
    setLoading(true);
    setPairs(null);
    setStatus({ msg: `Computing similar pairs (${predictionFilter})…`, type: "info" });
    try {
      const res = await fetchSimilarPairs(sessionId, threshold, predictionFilter);
      setPairs(res.pairs);
      setOk(`Found ${res.num_pairs} similar pairs (${predictionFilter}).`);
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

  async function handleReset() {
    if (sessionId) await deleteSession(sessionId).catch(() => {});
    setSessionId(null); setModelData(null); setDatasetMeta(null);
    setSummary(null); setPairs(null); setClusterData(null);
    setStep("upload-model"); setStatus(null);
    setPredictionFilter("all");
  }

  return (
    <div className="app">
      <Header step={step} sessionId={sessionId} onReset={handleReset} />
      <main className="main">
        {status && <StatusBar {...status} />}
        <div className="columns">
          <Sidebar
            step={step} loading={loading} sessionId={sessionId}
            datasetMeta={datasetMeta} labelColumn={labelColumn}
            threshold={threshold} predictionFilter={predictionFilter}
            onLabelColumnChange={setLabelColumn}
            onThresholdChange={setThreshold}
            onPredictionFilterChange={setPredictionFilter}
            onModelFile={handleModelFile}
            onDatasetFile={handleDatasetFile}
            onRunInference={handleRunInference}
            onFindPairs={handleFindPairs}
            onClusterPlot={handleClusterPlot}
          />
          <div className="col-right">
            {modelData    && <ModelInfo data={modelData} />}
            {summary      && <SummaryPanel summary={summary} />}
            {pairs        && <PairsPanel pairs={pairs} />}
            {clusterData  && (
              <ClusterPlot
                points={clusterData.points}
                method={clusterData.method}
              />
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
    </div>
  );
}