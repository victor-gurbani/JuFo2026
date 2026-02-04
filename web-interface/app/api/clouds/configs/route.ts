import { NextResponse } from "next/server";
import { readdir, readFile } from "node:fs/promises";
import { existsSync } from "node:fs";
import { join, resolve } from "node:path";

export const runtime = "nodejs";

type ConfigGroup = {
  name: string;
  description?: string;
  include_composer?: string[];
  exclude_composer?: string[];
  include_title?: string[];
  exclude_title?: string[];
  composer_aliases?: Record<string, string[] | string>;
};

type ConfigFile = {
  version?: number;
  notes?: string[];
  groups?: Record<string, ConfigGroup>;
};

function repoRoot(): string {
  return resolve(process.cwd(), "..");
}

function configsDir(): string {
  return join(repoRoot(), "configs");
}

export async function GET(request: Request) {
  try {
    const url = new URL(request.url);
    const id = url.searchParams.get("id");

    const dir = configsDir();
    if (!existsSync(dir)) {
      return NextResponse.json({ configs: [] });
    }

    const files = (await readdir(dir)).filter((f) => f.toLowerCase().endsWith(".json"));

    const loadOne = async (filename: string) => {
      const fullPath = join(dir, filename);
      const text = await readFile(fullPath, "utf-8");
      const parsed = JSON.parse(text) as ConfigFile;
      const groupsObj = parsed.groups ?? {};
      const groups = Object.entries(groupsObj).map(([name, g]) => ({
        name,
        description: g.description ?? "",
        include_composer: g.include_composer ?? [],
        exclude_composer: g.exclude_composer ?? [],
        include_title: g.include_title ?? [],
        exclude_title: g.exclude_title ?? [],
        composer_aliases: g.composer_aliases ?? undefined,
      }));
      return {
        id: filename,
        notes: parsed.notes ?? [],
        groupCount: groups.length,
        groups,
      };
    };

    if (id) {
      if (!files.includes(id)) {
        return NextResponse.json({ error: `Unknown config: ${id}` }, { status: 404 });
      }
      const one = await loadOne(id);
      return NextResponse.json(one);
    }

    const configs = await Promise.all(files.map(loadOne));
    return NextResponse.json({ configs });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Failed to load configs";
    return NextResponse.json({ error: message }, { status: 500 });
  }
}
