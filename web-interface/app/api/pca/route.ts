import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import { existsSync, statSync } from "node:fs";
import { basename, join, resolve } from "node:path";
import Papa from "papaparse";

type PcaPoint = {
  dim1: number;
  dim2: number;
  dim3: number;
  composer_label: string;
  title: string;
  mxl_path?: string;
  mxl_abs_path?: string;
};

type PcaCacheEntry = {
  mtimeMs: number;
  pcaData: PcaPoint[];
  source: string;
};

function getRepoRoot(): string {
  // In dev, process.cwd() is typically web-interface/. Repo root is one level up.
  return resolve(process.cwd(), "..");
}

function pickCachePath(repoRoot: string): string | null {
  const envPath = process.env.JUFO_PCA_CACHE_CSV;
  if (envPath && existsSync(envPath)) {
    return envPath;
  }
  const candidates = [
    join(repoRoot, "data", "embeddings", "pdmx_projected_cache.csv"),
    join(repoRoot, "data", "embeddings", "pca_embedding_cache.csv"),
  ];
  return candidates.find((p) => existsSync(p)) ?? null;
}

export async function GET() {
  try {
    const repoRoot = getRepoRoot();
    const cachePath = pickCachePath(repoRoot);
    if (cachePath) {
      const source = basename(cachePath);
      const stat = statSync(cachePath);

      const globalAny = globalThis as unknown as { __jufoPcaCache?: Record<string, PcaCacheEntry> };
      if (!globalAny.__jufoPcaCache) {
        globalAny.__jufoPcaCache = {};
      }

      const existing = globalAny.__jufoPcaCache[cachePath];
      if (existing && existing.mtimeMs === stat.mtimeMs) {
        return NextResponse.json({ pcaData: existing.pcaData, source: existing.source, cached: true });
      }

      const csvText = await readFile(cachePath, "utf-8");
      const parsed = Papa.parse<Record<string, string>>(csvText, {
        header: true,
        skipEmptyLines: true,
      });
      if (parsed.errors.length) {
        throw new Error(parsed.errors[0]?.message ?? "Failed to parse embedding cache CSV");
      }

      const pcaData: PcaPoint[] = parsed.data
        .filter((row) => row && row.dim1 !== undefined)
        .map((row) => ({
          dim1: Number.parseFloat(String(row.dim1)),
          dim2: Number.parseFloat(String(row.dim2)),
          dim3: Number.parseFloat(String(row.dim3)),
          composer_label: row.composer_label ?? "Unknown",
          title: row.title ?? "Untitled",
          mxl_path: row.mxl_path ?? "",
          mxl_abs_path: row.mxl_abs_path ?? "",
        }))
        .filter((row) => Number.isFinite(row.dim1) && Number.isFinite(row.dim2) && Number.isFinite(row.dim3));

      globalAny.__jufoPcaCache[cachePath] = { mtimeMs: stat.mtimeMs, pcaData, source };
      return NextResponse.json({ pcaData, source, cached: false });
    }

    // Fallback: keep dev server usable even without data present.
    const mockPcaData = [
      { dim1: -2.5, dim2: 1.2, dim3: 0.8, composer_label: "Bach", title: "Prelude in C Major" },
      { dim1: 0.2, dim2: -0.5, dim3: 1.3, composer_label: "Mozart", title: "Piano Sonata K. 545" },
      { dim1: 1.8, dim2: -1.2, dim3: -0.5, composer_label: "Chopin", title: "Nocturne Op. 9 No. 2" },
      { dim1: 2.8, dim2: 0.8, dim3: -1.2, composer_label: "Debussy", title: "Clair de Lune" },
    ];

    return NextResponse.json({ pcaData: mockPcaData, source: "mock" });
  } catch (error) {
    console.error("Error loading PCA data:", error);
    return NextResponse.json(
      { error: "Failed to load PCA data" },
      { status: 500 }
    );
  }
}
