import { useState } from "react";
import { Upload } from "lucide-react";

interface Props {
  label: string;
  accept: string;
  onFile: (file: File) => void;
  disabled?: boolean;
}

export default function DropZone({ label, accept, onFile, disabled }: Props) {
  const [dragging, setDragging] = useState(false);
  const [fileName, setFileName] = useState<string | null>(null);

  function handle(file: File) {
    setFileName(file.name);
    onFile(file);
  }

  return (
    <label
      className={`drop-zone ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => { e.preventDefault(); if (!disabled) setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        if (!disabled && e.dataTransfer.files[0]) handle(e.dataTransfer.files[0]);
      }}
    >
      <input
        type="file"
        accept={accept}
        disabled={disabled}
        style={{ display: "none" }}
        onChange={(e) => e.target.files?.[0] && handle(e.target.files[0])}
      />
      <Upload size={20} />
      <span>{fileName ?? label}</span>
    </label>
  );
}