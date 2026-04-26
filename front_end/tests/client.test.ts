import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  uploadModel,
  uploadDataset,
  runInference,
  fetchSimilarPairs,
  fetchClusterPlot,
  fetchIncorrectRecords,
  fetchLayerDeviation,
  deleteSession,
} from "../api/client";

const API_BASE = "http://localhost:8000";

interface FetchCall {
  url: string;
  init?: RequestInit;
}

let fetchCalls: FetchCall[] = [];

function mockFetchOk<T>(payload: T) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    fetchCalls.push({ url, init });
    return new Response(JSON.stringify(payload), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    });
  });
}

function mockFetchErr(detail: string, status = 500) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    fetchCalls.push({ url, init });
    return new Response(JSON.stringify({ detail }), {
      status,
      statusText: "Internal Server Error",
      headers: { "Content-Type": "application/json" },
    });
  });
}

beforeEach(() => {
  fetchCalls = [];
});

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("api/client - successful responses", () => {
  it("uploadModel POSTs to /upload/model with FormData", async () => {
    const payload = { session_id: "sess-1", model_data: { format: "keras" } };
    vi.stubGlobal("fetch", mockFetchOk(payload));

    const file = new File(["x"], "m.keras");
    const res = await uploadModel(file);

    expect(res).toEqual(payload);
    expect(fetchCalls[0].url).toBe(`${API_BASE}/upload/model`);
    expect(fetchCalls[0].init?.method).toBe("POST");
    expect(fetchCalls[0].init?.body).toBeInstanceOf(FormData);
  });

  it("uploadDataset includes the session_id and label_column in the URL", async () => {
    vi.stubGlobal("fetch", mockFetchOk({ filename: "x.csv", num_records: 5 }));

    await uploadDataset("sess-1", new File(["x"], "x.csv"), "label");

    expect(fetchCalls[0].url).toContain("session_id=sess-1");
    expect(fetchCalls[0].url).toContain("label_column=label");
  });

  it("uploadDataset omits label_column when not provided", async () => {
    vi.stubGlobal("fetch", mockFetchOk({ filename: "x.csv", num_records: 5 }));

    await uploadDataset("sess-1", new File(["x"], "x.csv"));

    expect(fetchCalls[0].url).not.toContain("label_column=");
  });

  it("runInference POSTs JSON with the session_id", async () => {
    const payload = { summary: { total_records: 1, has_labels: false } };
    vi.stubGlobal("fetch", mockFetchOk(payload));

    const res = await runInference("sess-1");

    expect(res).toEqual(payload);
    expect(fetchCalls[0].init?.headers).toEqual({
      "Content-Type": "application/json",
    });
    expect(JSON.parse(fetchCalls[0].init?.body as string)).toEqual({
      session_id: "sess-1",
    });
  });

  it("fetchSimilarPairs sends threshold and filter in the body", async () => {
    vi.stubGlobal("fetch", mockFetchOk({ pairs: [], num_pairs: 0 }));

    await fetchSimilarPairs("sess-1", 0.25, "incorrect");

    const body = JSON.parse(fetchCalls[0].init?.body as string);
    expect(body).toEqual({
      session_id: "sess-1",
      threshold: 0.25,
      filter: "incorrect",
    });
  });

  it("fetchClusterPlot returns the parsed payload", async () => {
    const payload = { method: "UMAP", points: [] };
    vi.stubGlobal("fetch", mockFetchOk(payload));

    const res = await fetchClusterPlot("sess-1", "all");

    expect(res).toEqual(payload);
  });

  it("fetchIncorrectRecords uses GET (no method override)", async () => {
    vi.stubGlobal("fetch", mockFetchOk({ records: [], total: 0 }));
    await fetchIncorrectRecords("sess-1");
    // The default for fetch when no init.method is provided is GET.
    expect(fetchCalls[0].init?.method).toBeUndefined();
    expect(fetchCalls[0].url).toContain("session_id=sess-1");
  });

  it("fetchLayerDeviation includes record_id in the body", async () => {
    vi.stubGlobal(
      "fetch",
      mockFetchOk({
        record_id: "r_1",
        true_label: 0,
        predicted_label: 1,
        layer_names: [],
        true_label_deviations: [],
        predicted_deviations: [],
      })
    );

    await fetchLayerDeviation("sess-1", "r_1");

    const body = JSON.parse(fetchCalls[0].init?.body as string);
    expect(body).toEqual({ session_id: "sess-1", record_id: "r_1" });
  });

  it("deleteSession issues a DELETE request to the session URL", async () => {
    vi.stubGlobal("fetch", mockFetchOk({}));
    await deleteSession("sess-1");
    expect(fetchCalls[0].url).toBe(`${API_BASE}/session/sess-1`);
    expect(fetchCalls[0].init?.method).toBe("DELETE");
  });
});

describe("api/client - error handling", () => {
  it("throws an Error with the API's detail message", async () => {
    vi.stubGlobal("fetch", mockFetchErr("Model not found", 404));
    await expect(runInference("sess-1")).rejects.toThrow("Model not found");
  });

  it("falls back to statusText if the error body isn't JSON", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        new Response("plain-text-error", {
          status: 500,
          statusText: "Boom",
          headers: { "Content-Type": "text/plain" },
        })
      )
    );
    await expect(runInference("sess-1")).rejects.toThrow("Boom");
  });
});
