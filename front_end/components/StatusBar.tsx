import { AlertCircle } from "lucide-react";
import type { StatusMessage } from "../types";

export default function StatusBar({ msg, type }: StatusMessage) {
  return (
    <div className={`status-bar status-${type}`}>
      <AlertCircle size={14} />
      <span>{msg}</span>
    </div>
  );
}