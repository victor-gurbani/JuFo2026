"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import StylisticSpace from "../../components/StylisticSpace";

type Axes = "canonical" | "refit" | "both";

type FeatureCacheItem = { id: string; sizeBytes: number };

type ConfigGroup = {
  name: string;
  description?: string;
  include_composer?: string[];
  exclude_composer?: string[];
  include_title?: string[];
  exclude_title?: string[];
};

type ConfigFile = {
  id: string;
  notes?: string[];
  groupCount?: number;
  groups: ConfigGroup[];
};

type CloudsJobOutputs = {
  manifestUrl?: string;
  canonical?: { htmlUrl?: string; jsonUrl?: string; subsetCsvUrl?: string };
  refit?: { htmlUrl?: string; jsonUrl?: string; subsetCsvUrl?: string };
  outDirUrl: string;
};

type CloudsJob = {
  id: string;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  status: "queued" | "running" | "done" | "error";
  error?: string;
  outputs?: CloudsJobOutputs;
};

function linesToArray(text: string): string[] {
  return text
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter(Boolean);
}

export default function CloudsPage() {
  const [featureCaches, setFeatureCaches] = useState<FeatureCacheItem[]>([]);
  const [selectedFeatureCacheId, setSelectedFeatureCacheId] = useState<string>("");

  const [configs, setConfigs] = useState<Array<{ id: string; groupCount: number; groups: ConfigGroup[]; notes?: string[] }>>([]);
  const [selectedConfigId, setSelectedConfigId] = useState<string>("");
  const [selectedGroup, setSelectedGroup] = useState<string>("");

  const [axes, setAxes] = useState<Axes>("both");
  const [seed, setSeed] = useState<number>(42);
  const [maxPerComposer, setMaxPerComposer] = useState<number>(200);
  const [limit, setLimit] = useState<number>(0);
  const [label, setLabel] = useState<string>("");
  const [writeSubsetCsv, setWriteSubsetCsv] = useState<boolean>(false);

  const [includeComposer, setIncludeComposer] = useState<string>("");
  const [excludeComposer, setExcludeComposer] = useState<string>("");
  const [includeTitle, setIncludeTitle] = useState<string>("");
  const [excludeTitle, setExcludeTitle] = useState<string>("");

  const [job, setJob] = useState<CloudsJob | null>(null);
  const [logLines, setLogLines] = useState<string[]>([]);
  const [logNext, setLogNext] = useState<number>(0);
  const [status, setStatus] = useState<string | null>(null);
  const [view, setView] = useState<"canonical" | "refit">("canonical");

  useEffect(() => {
    fetch("/api/clouds/feature-caches")
      .then((r) => r.json())
      .then((p) => {
        const items = (p.caches ?? []) as FeatureCacheItem[];
        setFeatureCaches(items);
        if (!selectedFeatureCacheId && items.length > 0) {
          setSelectedFeatureCacheId(items[0].id);
        }
      })
      .catch(() => {
        setFeatureCaches([]);
      });
  }, []);

  useEffect(() => {
    fetch("/api/clouds/configs")
      .then((r) => r.json())
      .then((p) => {
        const cfgs = (p.configs ?? []) as ConfigFile[];
        setConfigs(cfgs.map((c) => ({ id: c.id, groupCount: c.groupCount ?? c.groups.length, groups: c.groups, notes: c.notes })));
        if (!selectedConfigId) {
          setSelectedConfigId("");
        }
      })
      .catch(() => setConfigs([]));
  }, []);

  const selectedConfig = useMemo(() => configs.find((c) => c.id === selectedConfigId) ?? null, [configs, selectedConfigId]);

  useEffect(() => {
    if (!selectedConfig) {
      setSelectedGroup("");
      return;
    }
    if (selectedGroup && selectedConfig.groups.some((g) => g.name === selectedGroup)) {
      return;
    }
    setSelectedGroup(selectedConfig.groups[0]?.name ?? "");
  }, [selectedConfigId]);

  const startJob = async () => {
    if (!selectedFeatureCacheId) {
      setStatus("Select a feature cache first.");
      return;
    }

    setStatus("Starting job…");
    setLogLines([]);
    setLogNext(0);
    setJob(null);

    const body = {
      featureCacheId: selectedFeatureCacheId,
      axes,
      seed: Number.isFinite(seed) ? seed : undefined,
      maxPerComposer: maxPerComposer > 0 ? maxPerComposer : undefined,
      limit: limit > 0 ? limit : undefined,
      label: label.trim() || undefined,
      writeSubsetCsv,
      configId: selectedConfigId || undefined,
      group: selectedConfigId ? (selectedGroup || undefined) : undefined,
      includeComposer: linesToArray(includeComposer),
      excludeComposer: linesToArray(excludeComposer),
      includeTitle: linesToArray(includeTitle),
      excludeTitle: linesToArray(excludeTitle),
    };

    try {
      const res = await fetch("/api/clouds/jobs", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const payload = await res.json();
      if (!res.ok) {
        throw new Error(payload.error || "Failed to start job");
      }
      setJob(payload.job as CloudsJob);
      setStatus("Job running…");
    } catch (e) {
      setStatus(e instanceof Error ? e.message : "Failed to start job");
    }
  };

  const cancel = async () => {
    if (!job) return;
    try {
      await fetch(`/api/clouds/jobs/${job.id}/cancel`, { method: "POST" });
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    if (!job) return;
    if (job.status === "done" || job.status === "error") return;

    const interval = window.setInterval(async () => {
      try {
        const sRes = await fetch(`/api/clouds/jobs/${job.id}`);
        if (sRes.ok) {
          const sPayload = await sRes.json();
          setJob(sPayload.job as CloudsJob);
        }

        const lRes = await fetch(`/api/clouds/jobs/${job.id}/logs?since=${logNext}`);
        if (lRes.ok) {
          const lPayload = await lRes.json();
          const lines = (lPayload.lines ?? []) as string[];
          const next = (lPayload.next ?? logNext) as number;
          if (lines.length > 0) {
            setLogLines((prev) => [...prev, ...lines]);
          }
          setLogNext(next);
        }
      } catch {
        // ignore transient polling errors
      }
    }, 700);

    return () => window.clearInterval(interval);
  }, [job?.id, job?.status, logNext]);

  useEffect(() => {
    if (!job) return;
    if (job.status === "done") setStatus("Done.");
    if (job.status === "error") setStatus(job.error || "Job failed.");
  }, [job?.status]);

  const canonicalJson = job?.outputs?.canonical?.jsonUrl;
  const refitJson = job?.outputs?.refit?.jsonUrl;

  useEffect(() => {
    if (axes === "canonical") setView("canonical");
    else if (axes === "refit") setView("refit");
    else {
      if (view === "canonical" && !canonicalJson && refitJson) setView("refit");
      if (view === "refit" && !refitJson && canonicalJson) setView("canonical");
    }
  }, [axes, canonicalJson, refitJson]);

  const fmtBytes = (n: number) => {
    const units = ["B", "KB", "MB", "GB"];
    let v = n;
    let i = 0;
    while (v >= 1024 && i < units.length - 1) {
      v /= 1024;
      i++;
    }
    return `${v.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
  };

  return (
    <main className="min-h-screen bg-zinc-950 text-white">
      <div className="mx-auto max-w-7xl px-6 py-10">
        <div className="flex items-center justify-between gap-6">
          <div>
            <h1 className="text-2xl font-semibold">Subset Clouds</h1>
            <p className="mt-1 text-sm text-zinc-300">
              Run <span className="font-mono text-zinc-200">clouds_from_feature_cache.py</span> from the UI (local-only).
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Link className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-zinc-200 hover:bg-white/10" href="/">
              Back to main
            </Link>
          </div>
        </div>

        <div className="mt-8 grid grid-cols-1 gap-6 lg:grid-cols-3">
          <section className="rounded-3xl border border-white/10 bg-white/5 p-6">
            <h2 className="text-sm font-semibold text-zinc-200">Inputs</h2>

            <label className="mt-4 block text-xs text-zinc-300">Feature cache</label>
            <select
              className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
              value={selectedFeatureCacheId}
              onChange={(e) => setSelectedFeatureCacheId(e.target.value)}
            >
              {featureCaches.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} ({fmtBytes(c.sizeBytes)})
                </option>
              ))}
            </select>

            <label className="mt-4 block text-xs text-zinc-300">Config</label>
            <select
              className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
              value={selectedConfigId}
              onChange={(e) => setSelectedConfigId(e.target.value)}
            >
              <option value="">(none)</option>
              {configs.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.id} ({c.groupCount} groups)
                </option>
              ))}
            </select>

            <label className="mt-4 block text-xs text-zinc-300">Group</label>
            <select
              className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
              value={selectedGroup}
              onChange={(e) => setSelectedGroup(e.target.value)}
              disabled={!selectedConfigId || !selectedConfig}
            >
              {(selectedConfig?.groups ?? []).map((g) => (
                <option key={g.name} value={g.name}>
                  {g.name}
                </option>
              ))}
            </select>
            {selectedConfig?.groups.find((g) => g.name === selectedGroup)?.description ? (
              <p className="mt-2 text-xs text-zinc-400">
                {selectedConfig.groups.find((g) => g.name === selectedGroup)?.description}
              </p>
            ) : null}

            <div className="mt-6 grid grid-cols-2 gap-3">
              <div>
                <label className="block text-xs text-zinc-300">Axes</label>
                <select
                  className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
                  value={axes}
                  onChange={(e) => setAxes(e.target.value as Axes)}
                >
                  <option value="both">both</option>
                  <option value="canonical">canonical</option>
                  <option value="refit">refit</option>
                </select>
              </div>
              <div>
                <label className="block text-xs text-zinc-300">Seed</label>
                <input
                  className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
                  value={seed}
                  onChange={(e) => setSeed(parseInt(e.target.value, 10) || 0)}
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-300">Max per composer</label>
                <input
                  className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
                  value={maxPerComposer}
                  onChange={(e) => setMaxPerComposer(parseInt(e.target.value, 10) || 0)}
                />
              </div>
              <div>
                <label className="block text-xs text-zinc-300">Global limit</label>
                <input
                  className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
                  value={limit}
                  onChange={(e) => setLimit(parseInt(e.target.value, 10) || 0)}
                />
              </div>
            </div>

            <label className="mt-4 block text-xs text-zinc-300">Label (optional)</label>
            <input
              className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-sm"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g., spanish_nationalist"
            />

            <label className="mt-4 flex items-center gap-2 text-xs text-zinc-300">
              <input type="checkbox" checked={writeSubsetCsv} onChange={(e) => setWriteSubsetCsv(e.target.checked)} />
              Write subset CSV
            </label>

            <div className="mt-6 rounded-2xl border border-white/10 bg-zinc-950/60 p-4">
              <div className="text-xs font-semibold text-zinc-200">Custom regex filters (optional)</div>
              <p className="mt-1 text-[11px] text-zinc-400">One regex per line. These are added on top of the selected group.</p>

              <label className="mt-3 block text-[11px] text-zinc-300">Include composer</label>
              <textarea
                className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
                rows={2}
                value={includeComposer}
                onChange={(e) => setIncludeComposer(e.target.value)}
                placeholder="e.g. ^Debussy$\nRavel"
              />

              <label className="mt-3 block text-[11px] text-zinc-300">Exclude composer</label>
              <textarea
                className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
                rows={2}
                value={excludeComposer}
                onChange={(e) => setExcludeComposer(e.target.value)}
                placeholder="e.g. Bach|Mozart"
              />

              <label className="mt-3 block text-[11px] text-zinc-300">Include title</label>
              <textarea
                className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
                rows={2}
                value={includeTitle}
                onChange={(e) => setIncludeTitle(e.target.value)}
                placeholder="e.g. nocturne"
              />

              <label className="mt-3 block text-[11px] text-zinc-300">Exclude title</label>
              <textarea
                className="mt-2 w-full rounded-xl border border-white/10 bg-zinc-950 px-3 py-2 text-xs"
                rows={2}
                value={excludeTitle}
                onChange={(e) => setExcludeTitle(e.target.value)}
                placeholder="e.g. arrangement"
              />
            </div>

            <div className="mt-6 flex gap-3">
              <button
                className="flex-1 rounded-xl bg-indigo-600 px-4 py-2 text-sm font-medium text-white hover:bg-indigo-500 disabled:opacity-50"
                onClick={startJob}
                disabled={job?.status === "running"}
              >
                Generate
              </button>
              <button
                className="rounded-xl border border-white/10 bg-white/5 px-4 py-2 text-sm text-zinc-200 hover:bg-white/10 disabled:opacity-50"
                onClick={cancel}
                disabled={!job || job.status !== "running"}
              >
                Cancel
              </button>
            </div>

            {status ? <p className="mt-4 text-xs text-zinc-300">{status}</p> : null}
          </section>

          <section className="rounded-3xl border border-white/10 bg-white/5 p-6 lg:col-span-2">
            <div className="flex items-center justify-between gap-3">
              <h2 className="text-sm font-semibold text-zinc-200">Output</h2>
              <div className="flex items-center gap-2">
                <button
                  className={`rounded-xl px-3 py-1.5 text-xs ${view === "canonical" ? "bg-white/10" : "bg-white/5 hover:bg-white/10"}`}
                  onClick={() => setView("canonical")}
                  disabled={!canonicalJson}
                >
                  Canonical
                </button>
                <button
                  className={`rounded-xl px-3 py-1.5 text-xs ${view === "refit" ? "bg-white/10" : "bg-white/5 hover:bg-white/10"}`}
                  onClick={() => setView("refit")}
                  disabled={!refitJson}
                >
                  Refit
                </button>
              </div>
            </div>

            <div className="mt-4 h-[520px]">
              {view === "canonical" && canonicalJson ? (
                <StylisticSpace dataUrl={`${canonicalJson}?t=${Date.now()}`} />
              ) : null}
              {view === "refit" && refitJson ? (
                <StylisticSpace dataUrl={`${refitJson}?t=${Date.now()}`} />
              ) : null}
              {!canonicalJson && !refitJson ? (
                <div className="flex h-full items-center justify-center rounded-3xl border border-white/10 bg-white/5 p-8 text-sm text-zinc-300">
                  No plot yet.
                </div>
              ) : null}
            </div>

            {job?.outputs ? (
              <div className="mt-4 grid grid-cols-1 gap-3 text-xs text-zinc-300 md:grid-cols-2">
                <div className="rounded-2xl border border-white/10 bg-zinc-950/60 p-4">
                  <div className="font-semibold text-zinc-200">Links</div>
                  <div className="mt-2 flex flex-col gap-1">
                    {job.outputs.manifestUrl ? (
                      <a className="text-indigo-300 hover:underline" href={job.outputs.manifestUrl} target="_blank" rel="noreferrer">
                        manifest
                      </a>
                    ) : null}
                    {job.outputs.canonical?.htmlUrl ? (
                      <a className="text-indigo-300 hover:underline" href={job.outputs.canonical.htmlUrl} target="_blank" rel="noreferrer">
                        canonical html
                      </a>
                    ) : null}
                    {job.outputs.canonical?.subsetCsvUrl ? (
                      <a className="text-indigo-300 hover:underline" href={job.outputs.canonical.subsetCsvUrl} target="_blank" rel="noreferrer">
                        canonical subset csv
                      </a>
                    ) : null}
                    {job.outputs.refit?.htmlUrl ? (
                      <a className="text-indigo-300 hover:underline" href={job.outputs.refit.htmlUrl} target="_blank" rel="noreferrer">
                        refit html
                      </a>
                    ) : null}
                    {job.outputs.refit?.subsetCsvUrl ? (
                      <a className="text-indigo-300 hover:underline" href={job.outputs.refit.subsetCsvUrl} target="_blank" rel="noreferrer">
                        refit subset csv
                      </a>
                    ) : null}
                  </div>
                </div>

                <div className="rounded-2xl border border-white/10 bg-zinc-950/60 p-4">
                  <div className="font-semibold text-zinc-200">Job</div>
                  <div className="mt-2 flex flex-col gap-1">
                    <div>id: <span className="font-mono">{job.id}</span></div>
                    <div>status: {job.status}</div>
                    {job.error ? <div className="text-rose-300">error: {job.error}</div> : null}
                  </div>
                </div>
              </div>
            ) : null}

            <div className="mt-4 rounded-2xl border border-white/10 bg-zinc-950/60 p-4">
              <div className="text-xs font-semibold text-zinc-200">Logs</div>
              <pre className="mt-2 max-h-56 overflow-auto whitespace-pre-wrap text-[11px] leading-snug text-zinc-300">
                {logLines.join("\n")}
              </pre>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}
