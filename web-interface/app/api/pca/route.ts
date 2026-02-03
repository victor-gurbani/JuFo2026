import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join } from "node:path";
import Papa from "papaparse";

export async function GET() {
  try {
    // Prefer the embedding cache computed by src/embedding_cache.py (repo root).
    const cachePath = join(process.cwd(), "..", "data", "embeddings", "pca_embedding_cache.csv");
    if (existsSync(cachePath)) {
      const csvText = await readFile(cachePath, "utf-8");
      const parsed = Papa.parse<Record<string, string>>(csvText, {
        header: true,
        skipEmptyLines: true,
      });
      if (parsed.errors.length) {
        throw new Error(parsed.errors[0]?.message ?? "Failed to parse embedding cache CSV");
      }

      const pcaData = parsed.data
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

      return NextResponse.json({ pcaData, source: "cache" });
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
