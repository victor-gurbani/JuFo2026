
import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const folder = searchParams.get("folder");

  if (!folder) {
    return NextResponse.json({ error: "Folder required" }, { status: 400 });
  }

  // Safe list of allowed folders to prevent directory traversal
  const allowedFolders = ["harmonic", "melodic", "rhythmic", "significance", "embeddings", "highlights"];
  if (!allowedFolders.includes(folder)) {
    return NextResponse.json({ error: "Invalid folder" }, { status: 403 });
  }

  const figuresPath = path.join(process.cwd(), "..", "figures", folder);
  
  try {
    const files = await fs.promises.readdir(figuresPath);
    const images = files
      .filter(file => file.endsWith(".png") || file.endsWith(".jpg") || file.endsWith(".jpeg") || file.endsWith(".svg"))
      .map(file => `/figures/${folder}/${file}`);
    
    return NextResponse.json({ images });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ images: [] });
  }
}
