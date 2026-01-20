import { NextResponse } from "next/server";
import { execFile } from "node:child_process";
import { existsSync } from "node:fs";
import { mkdir } from "node:fs/promises";
import { promisify } from "node:util";
import path from "node:path";

const execFileAsync = promisify(execFile);

type AnalyzePayload = {
  mxlPath?: string;
  title?: string;
  composer?: string;
};

export async function POST(request: Request) {
  try {
    const payload = (await request.json()) as AnalyzePayload;
    if (!payload.mxlPath) {
      return NextResponse.json({ error: "Missing mxlPath" }, { status: 400 });
    }

    const appRoot = process.cwd();
    const repoRoot = path.resolve(appRoot, "..");
    const scriptPath = path.join(repoRoot, "src", "highlight_pca_piece.py");
    const outputDir = path.join(appRoot, "public", "temp");
    const outputPath = path.join(outputDir, "latest_analysis.html");

    await mkdir(outputDir, { recursive: true });

    const resolveCandidate = (candidate: string) => (existsSync(candidate) ? candidate : "");
    const rawPath = payload.mxlPath;
    const candidates: string[] = [];

    if (path.isAbsolute(rawPath)) {
      candidates.push(rawPath);
    } else {
      candidates.push(path.join(repoRoot, rawPath));
      candidates.push(path.join(repoRoot, "15571083", rawPath));
      const mxlIndex = rawPath.indexOf("/mxl/");
      if (mxlIndex >= 0) {
        candidates.push(path.join(repoRoot, "15571083", rawPath.slice(mxlIndex + 1)));
        candidates.push(path.join(repoRoot, "15571083", rawPath.slice(mxlIndex + 5)));
      }
    }

    const mxlPath = candidates.map(resolveCandidate).find(Boolean);
    if (!mxlPath) {
      return NextResponse.json({ error: `MusicXML not found: ${rawPath}` }, { status: 400 });
    }

    const args = [
      scriptPath,
      mxlPath,
      "--output",
      outputPath,
    ];

    if (payload.composer) {
      args.push("--composer", payload.composer);
    }
    if (payload.title) {
      args.push("--title", payload.title);
    }

    await execFileAsync("python3", args, { cwd: repoRoot });

    return NextResponse.json({
      ok: true,
      jsonUrl: "/temp/latest_analysis.json",
      htmlUrl: "/temp/latest_analysis.html",
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Analysis failed";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
