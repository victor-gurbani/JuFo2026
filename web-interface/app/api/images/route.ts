import { NextResponse } from "next/server";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const folder = searchParams.get("folder");

  if (!folder) {
    return NextResponse.json({ error: "Folder required" }, { status: 400 });
  }

  // Safe list of allowed folders to prevent directory traversal
  const allowedFolders = ["harmonic", "melodic", "rhythmic", "significance", "embeddings", "highlights", "random_forest", "evolution", "architecture"];
  if (!allowedFolders.includes(folder)) {
    return NextResponse.json({ error: "Invalid folder" }, { status: 403 });
  }

  // Use process.cwd() to be resilient to standalone vs default outputs
  const appRoot = process.cwd();
  const figuresPath = path.join(appRoot, "public", "figures", folder);
  
  try {
    const files = await fs.promises.readdir(figuresPath);
    const images = files
      .filter(file => file.endsWith(".png") || file.endsWith(".jpg") || file.endsWith(".jpeg") || file.endsWith(".svg") || file.endsWith(".gif"))
      .map(file => `/figures/${folder}/${file}`);
    
    return NextResponse.json({ images });
  } catch (err) {
    console.error(err);
    return NextResponse.json({ images: [] });
  }
}
