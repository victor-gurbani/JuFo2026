import { NextResponse } from "next/server";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import Papa from "papaparse";

export async function GET() {
  try {
    // Path to the curated corpus CSV
    const csvPath = join(process.cwd(), "..", "data", "curated", "solo_piano_corpus.csv");
    
    const fileContent = await readFile(csvPath, "utf-8");
    const parsed = Papa.parse<Record<string, string>>(fileContent, {
      header: true,
      skipEmptyLines: true,
    });
    if (parsed.errors.length) {
      throw new Error(parsed.errors[0]?.message ?? "Failed to parse corpus CSV");
    }
    const records = parsed.data as Array<Record<string, string>>;
    
    // Transform the data to match our interface
    const corpusEntries = records.map((record) => ({
      composer: record.composer_label || record.composer_name || "Unknown",
      title: record.title || record.song_name || "Untitled",
      mxl_path: record.mxl_path || record.mxl_abs_path || "",
      rating: record.rating ? Number.parseFloat(record.rating) : null,
      instrument: record.instrument_names || record.instrument || "Piano",
    }));
    
    return NextResponse.json({ corpusEntries });
  } catch (error) {
    console.error("Error loading corpus:", error);
    return NextResponse.json(
      { error: "Failed to load corpus data" },
      { status: 500 }
    );
  }
}
