import { Cpu } from "lucide-react";
import type { ModelData } from "../types";

interface Props {
  data: ModelData;
}

export default function ModelInfo({ data }: Props) {
  return (
    <div className="panel">
      <h3 className="panel-title">
        <Cpu size={14} /> Model Architecture
      </h3>

      <div className="kv-grid">
        <span className="kv-key">Format</span>
        <span className="kv-val">{data.format.toUpperCase()}</span>

        <span className="kv-key">Layers</span>
        <span className="kv-val">{data.num_layers}</span>

        <span className="kv-key">Total params</span>
        <span className="kv-val">{data.total_params.toLocaleString()}</span>

        <span className="kv-key">Input shape</span>
        <span className="kv-val">[{data.input_shape.join(", ")}]</span>

        <span className="kv-key">Output shape</span>
        <span className="kv-val">[{data.output_shape.join(", ")}]</span>
      </div>

      <div className="layer-list">
        {data.layers.map((layer, i) => (
          <div
            key={i}
            className={`layer-row ${!layer.relevant_inference ? "layer-inactive" : ""}`}
          >
            <span className="layer-idx">{i}</span>
            <span className="layer-type">{layer.type}</span>
            <span className="layer-name">{layer.name}</span>
            {layer.activation && (
              <span className="layer-tag">{layer.activation}</span>
            )}
            {layer.num_neurons && (
              <span className="layer-tag neurons">{layer.num_neurons}n</span>
            )}
            {!layer.relevant_inference && (
              <span className="layer-tag inactive">skip</span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}