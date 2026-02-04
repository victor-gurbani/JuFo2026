import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { mkdirSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { join, resolve } from "node:path";

export type CloudsAxes = "canonical" | "refit" | "both";

export type CloudsJobRequest = {
  featureCachePath: string; // absolute path
  axes: CloudsAxes;
  canonicalModelCachePath: string; // absolute path
  seed?: number;
  maxPerComposer?: number;
  limit?: number;
  label?: string;
  writeSubsetCsv?: boolean;
  // predefined config path is allowed (repo-owned)
  configPath?: string; // absolute path
  group?: string;
  // Optional ephemeral group config (written into the job output directory).
  // Used for DIY composer selection without persisting server-side config files.
  ephemeralGroup?: {
    groupName: string;
    group: Record<string, unknown>;
  };
  // patterns are passed as CLI flags (for custom config upload handled client-side)
  includeComposer?: string[];
  excludeComposer?: string[];
  includeTitle?: string[];
  excludeTitle?: string[];
};

export type CloudsJobStatus = "queued" | "running" | "done" | "error";

export type CloudsJobOutputs = {
  manifestUrl?: string;
  canonical?: { htmlUrl?: string; jsonUrl?: string; subsetCsvUrl?: string };
  refit?: { htmlUrl?: string; jsonUrl?: string; subsetCsvUrl?: string };
  outDirUrl: string;
};

export type CloudsJob = {
  id: string;
  createdAt: string;
  startedAt?: string;
  endedAt?: string;
  status: CloudsJobStatus;
  request: CloudsJobRequest;
  outDirAbs: string;
  outDirPublicUrl: string;
  logs: string[];
  exitCode?: number;
  error?: string;
  outputs?: CloudsJobOutputs;
  process?: ChildProcessWithoutNullStreams;
};

type Registry = {
  jobs: Record<string, CloudsJob>;
};

function getRegistry(): Registry {
  const globalAny = globalThis as unknown as { __jufoCloudJobs?: Registry };
  if (!globalAny.__jufoCloudJobs) {
    globalAny.__jufoCloudJobs = { jobs: {} };
  }
  return globalAny.__jufoCloudJobs;
}

function nowIso(): string {
  return new Date().toISOString();
}

function newJobId(): string {
  const rand = Math.random().toString(16).slice(2, 10);
  return `clouds_${Date.now()}_${rand}`;
}

function appendLog(job: CloudsJob, line: string) {
  const trimmed = line.replace(/\r?\n$/, "");
  if (!trimmed) return;
  job.logs.push(trimmed);
  // cap memory
  if (job.logs.length > 4000) {
    job.logs.splice(0, job.logs.length - 4000);
  }
}

class LineSplitter {
  private buffer = "";
  constructor(private readonly onLine: (line: string) => void) {}
  push(chunk: Buffer) {
    this.buffer += chunk.toString("utf-8");
    const parts = this.buffer.split(/\r?\n/);
    this.buffer = parts.pop() ?? "";
    for (const line of parts) {
      this.onLine(line);
    }
  }
  flush() {
    if (this.buffer) {
      this.onLine(this.buffer);
      this.buffer = "";
    }
  }
}

function discoverOutputs(outDirAbs: string, outDirPublicUrl: string): CloudsJobOutputs {
  const outputs: CloudsJobOutputs = { outDirUrl: outDirPublicUrl };
  const files = readdirSync(outDirAbs);

  const pick = (suffix: string): string | undefined => {
    const match = files.find((f) => f.endsWith(suffix));
    return match ? `${outDirPublicUrl}/${match}` : undefined;
  };

  outputs.manifestUrl = pick("__run_manifest.json");
  outputs.canonical = {
    htmlUrl: pick("__canonical_axes_clouds.html"),
    jsonUrl: pick("__canonical_axes_clouds.json"),
    subsetCsvUrl: pick("__canonical_axes_subset.csv"),
  };
  outputs.refit = {
    htmlUrl: pick("__refit_axes_clouds.html"),
    jsonUrl: pick("__refit_axes_clouds.json"),
    subsetCsvUrl: pick("__refit_axes_subset.csv"),
  };

  return outputs;
}

export function getJob(jobId: string): CloudsJob | null {
  const registry = getRegistry();
  return registry.jobs[jobId] ?? null;
}

export function listJobs(): CloudsJob[] {
  const registry = getRegistry();
  return Object.values(registry.jobs).sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
}

export function startCloudsJob(request: CloudsJobRequest, opts: { repoRoot: string; appRoot: string }): CloudsJob {
  const { repoRoot, appRoot } = opts;
  const registry = getRegistry();

  const id = newJobId();
  const outDirAbs = resolve(appRoot, "public", "generated", "clouds", id);
  const outDirPublicUrl = `/generated/clouds/${id}`;

  mkdirSync(outDirAbs, { recursive: true });

  const job: CloudsJob = {
    id,
    createdAt: nowIso(),
    status: "queued",
    request,
    outDirAbs,
    outDirPublicUrl,
    logs: [],
  };
  registry.jobs[id] = job;

  const scriptPath = join(repoRoot, "src", "clouds_from_feature_cache.py");

  const args: string[] = [
    scriptPath,
    "--feature-cache",
    request.featureCachePath,
    "--axes",
    request.axes,
    "--canonical-model-cache",
    request.canonicalModelCachePath,
    "--outdir",
    outDirAbs,
  ];

  if (request.ephemeralGroup) {
    const safeName = String(request.ephemeralGroup.groupName || "diy")
      .replace(/[^a-zA-Z0-9_\-]+/g, "_")
      .slice(0, 64) || "diy";
    const tmpConfigPath = join(outDirAbs, "__ephemeral_config.json");
    const payload = {
      version: 1,
      notes: ["Ephemeral config generated by /clouds DIY composer selection."],
      groups: {
        [safeName]: request.ephemeralGroup.group,
      },
    };
    writeFileSync(tmpConfigPath, JSON.stringify(payload, null, 2), "utf-8");
    args.push("--config", tmpConfigPath, "--group", safeName);
  } else if (request.configPath && request.group) {
    args.push("--config", request.configPath, "--group", request.group);
  }

  const pushMany = (flag: string, values?: string[]) => {
    for (const v of values ?? []) {
      const text = String(v).trim();
      if (text) args.push(flag, text);
    }
  };

  pushMany("--include-composer", request.includeComposer);
  pushMany("--exclude-composer", request.excludeComposer);
  pushMany("--include-title", request.includeTitle);
  pushMany("--exclude-title", request.excludeTitle);

  if (typeof request.maxPerComposer === "number") {
    args.push("--max-per-composer", String(request.maxPerComposer));
  }
  if (typeof request.limit === "number") {
    args.push("--limit", String(request.limit));
  }
  if (typeof request.seed === "number") {
    args.push("--seed", String(request.seed));
  }
  if (request.label) {
    args.push("--label", request.label);
  }
  if (request.writeSubsetCsv) {
    args.push("--write-subset-csv");
  }

  job.status = "running";
  job.startedAt = nowIso();

  appendLog(job, `[job] starting ${id}`);
  appendLog(job, `[job] outdir=${outDirAbs}`);

  const child = spawn("python3", args, { cwd: repoRoot });
  job.process = child;

  const outSplit = new LineSplitter((line) => appendLog(job, line));
  const errSplit = new LineSplitter((line) => appendLog(job, `[stderr] ${line}`));

  child.stdout.on("data", (chunk) => outSplit.push(chunk as Buffer));
  child.stderr.on("data", (chunk) => errSplit.push(chunk as Buffer));

  child.on("close", (code) => {
    outSplit.flush();
    errSplit.flush();

    job.exitCode = code ?? undefined;
    job.endedAt = nowIso();

    if (code === 0) {
      job.status = "done";
      try {
        job.outputs = discoverOutputs(outDirAbs, outDirPublicUrl);
      } catch (e) {
        job.outputs = { outDirUrl: outDirPublicUrl };
        job.error = e instanceof Error ? e.message : String(e);
      }
      appendLog(job, `[job] done exit=${code}`);
    } else {
      job.status = "error";
      job.error = `Process exited with code ${code}`;
      appendLog(job, `[job] error exit=${code}`);
    }

    job.process = undefined;
  });

  child.on("error", (err) => {
    job.status = "error";
    job.error = err instanceof Error ? err.message : String(err);
    job.endedAt = nowIso();
    appendLog(job, `[job] spawn error: ${job.error}`);
    job.process = undefined;
  });

  return job;
}

export function cancelJob(jobId: string): { ok: boolean; error?: string } {
  const job = getJob(jobId);
  if (!job) return { ok: false, error: `Unknown job: ${jobId}` };
  if (job.status !== "running" || !job.process) {
    return { ok: false, error: "Job is not running" };
  }

  try {
    job.process.kill("SIGTERM");
    appendLog(job, "[job] cancel requested (SIGTERM)");
    return { ok: true };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    appendLog(job, `[job] cancel failed: ${message}`);
    return { ok: false, error: message };
  }
}

export function fileStatSafe(path: string): { sizeBytes: number } | null {
  try {
    const st = statSync(path);
    return { sizeBytes: st.size };
  } catch {
    return null;
  }
}
