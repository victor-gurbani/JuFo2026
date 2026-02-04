import { NextResponse } from "next/server";
import { readdir } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";
import { fileStatSafe } from "@/lib/cloudJobs";

export const runtime = "nodejs";

type CacheItem = {
  id: string; // repo-relative path
  sizeBytes: number;
};

function repoRoot(): string {
  return resolve(process.cwd(), "..");
}

async function scanDir(dirAbs: string, repo: string): Promise<CacheItem[]> {
  if (!existsSync(dirAbs)) return [];
  const files = (await readdir(dirAbs)).filter((f) => f.toLowerCase().endsWith(".csv"));

  const items: CacheItem[] = [];
  for (const f of files) {
    const lower = f.toLowerCase();
    const looksLikeFeatureCache = lower.includes("feature") && lower.includes("cache");
    if (!looksLikeFeatureCache && !lower.includes("full_pdmx")) continue;

    const abs = join(dirAbs, f);
    const st = fileStatSafe(abs);
    if (!st) continue;

    const rel = abs.startsWith(repo) ? abs.slice(repo.length + 1) : f;
    items.push({ id: rel, sizeBytes: st.sizeBytes });
  }
  return items;
}

export async function GET() {
  try {
    const repo = repoRoot();
    const candidates = [join(repo, "data", "embeddings"), join(repo, "data", "features")];
    const results = (await Promise.all(candidates.map((d) => scanDir(d, repo)))).flat();

    // prefer embeddings directory first, then larger files first
    results.sort((a, b) => b.sizeBytes - a.sizeBytes);

    return NextResponse.json({ caches: results });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to list feature caches";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
