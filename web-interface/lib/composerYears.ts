export type ComposerYearMap = Record<string, number>;

export function normalizeComposerKey(value: string): string {
  return value
    .normalize("NFKD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/[^A-Za-z0-9]+/g, " ")
    .toLowerCase()
    .trim()
    .replace(/\s+/g, " ");
}

export function resolveComposerYear(map: ComposerYearMap | null | undefined, name: string): number | null {
  if (!map) return null;
  const direct = map[name];
  if (typeof direct === "number" && Number.isFinite(direct)) return direct;
  const normalized = map[normalizeComposerKey(name)];
  if (typeof normalized === "number" && Number.isFinite(normalized)) return normalized;
  return null;
}

export async function loadComposerYearMap(): Promise<ComposerYearMap> {
  // Loaded from a JSON file so it’s easy to edit/extend without touching code.
  // We store BOTH exact keys and normalized keys in the returned map.
  const res = await fetch("/data/composer_birth_years.json", { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Failed to load composer birth years: ${res.status}`);
  }
  const raw = (await res.json()) as Record<string, unknown>;
  const exact: Record<string, number> = {};
  for (const [k, v] of Object.entries(raw)) {
    const year = typeof v === "number" ? v : Number.parseFloat(String(v));
    if (!Number.isFinite(year)) continue;
    exact[String(k)] = year;
  }
  const normalized: Record<string, number> = {};
  for (const [name, year] of Object.entries(exact)) {
    normalized[normalizeComposerKey(name)] = year;
  }
  return { ...exact, ...normalized };
}
