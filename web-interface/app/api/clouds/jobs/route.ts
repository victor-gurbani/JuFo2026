import { NextResponse } from "next/server";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import {
  listJobs,
  startCloudsJob,
  type CloudsAxes,
  type CloudsJobRequest,
} from "@/lib/cloudJobs";

export const runtime = "nodejs";

function repoRoot(): string {
  return resolve(process.cwd(), "..");
}

function appRoot(): string {
  return resolve(process.cwd());
}

function resolveRepoPath(repo: string, relPath: string): string {
  const resolved = resolve(repo, relPath);
  if (!resolved.startsWith(repo + "/")) {
    throw new Error(`Path escapes repo root: ${relPath}`);
  }
  return resolved;
}

function resolveConfigPath(repo: string, id: string): string {
  if (id.includes("/") || id.includes("\\")) {
    throw new Error("Invalid config id");
  }
  const full = join(repo, "configs", id);
  if (!existsSync(full)) {
    throw new Error(`Config not found: ${id}`);
  }
  return full;
}

type StartJobBody = {
  featureCacheId: string; // repo-relative path, from /api/clouds/feature-caches
  axes?: CloudsAxes;
  canonicalModelCacheId?: string; // repo-relative path (optional)
  seed?: number;
  maxPerComposer?: number;
  limit?: number;
  label?: string;
  writeSubsetCsv?: boolean;

  configId?: string; // filename under /configs
  group?: string;

  includeComposer?: string[];
  excludeComposer?: string[];
  includeTitle?: string[];
  excludeTitle?: string[];
};

export async function GET() {
  return NextResponse.json({ jobs: listJobs().map((j) => ({
    id: j.id,
    createdAt: j.createdAt,
    startedAt: j.startedAt,
    endedAt: j.endedAt,
    status: j.status,
    error: j.error,
    outputs: j.outputs,
  })) });
}

export async function POST(request: Request) {
  try {
    const repo = repoRoot();
    const app = appRoot();

    const body = (await request.json()) as StartJobBody;

    if (!body?.featureCacheId || typeof body.featureCacheId !== "string") {
      return NextResponse.json({ error: "featureCacheId is required" }, { status: 400 });
    }

    const axes: CloudsAxes = (body.axes ?? "both") as CloudsAxes;
    if (!(["canonical", "refit", "both"] as const).includes(axes)) {
      return NextResponse.json({ error: `Invalid axes: ${String(body.axes)}` }, { status: 400 });
    }

    const featureCachePath = resolveRepoPath(repo, body.featureCacheId);
    if (!existsSync(featureCachePath)) {
      return NextResponse.json({ error: `Feature cache not found: ${body.featureCacheId}` }, { status: 400 });
    }

    const canonicalModelRel = body.canonicalModelCacheId ?? "data/embeddings/pca_embedding_cache.csv";
    const canonicalModelCachePath = resolveRepoPath(repo, canonicalModelRel);
    if (!existsSync(canonicalModelCachePath)) {
      return NextResponse.json(
        { error: `Canonical model cache not found: ${canonicalModelRel}` },
        { status: 400 },
      );
    }

    let configPath: string | undefined;
    if (body.configId) {
      configPath = resolveConfigPath(repo, body.configId);
    }

    const req: CloudsJobRequest = {
      featureCachePath,
      axes,
      canonicalModelCachePath,
      seed: typeof body.seed === "number" ? body.seed : undefined,
      maxPerComposer: typeof body.maxPerComposer === "number" ? body.maxPerComposer : undefined,
      limit: typeof body.limit === "number" ? body.limit : undefined,
      label: typeof body.label === "string" ? body.label : undefined,
      writeSubsetCsv: Boolean(body.writeSubsetCsv),
      configPath,
      group: typeof body.group === "string" ? body.group : undefined,
      includeComposer: Array.isArray(body.includeComposer) ? body.includeComposer : undefined,
      excludeComposer: Array.isArray(body.excludeComposer) ? body.excludeComposer : undefined,
      includeTitle: Array.isArray(body.includeTitle) ? body.includeTitle : undefined,
      excludeTitle: Array.isArray(body.excludeTitle) ? body.excludeTitle : undefined,
    };

    const job = startCloudsJob(req, { repoRoot: repo, appRoot: app });

    return NextResponse.json({ jobId: job.id, job });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to start job";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
