"use client";

import { useEffect, useMemo, useState } from "react";
import Papa from "papaparse";

export type CorpusEntry = {
  composer_name?: string;
  composer_label?: string;
  title?: string;
  song_name?: string;
  mxl_path?: string;
  mxl_abs_path?: string;
  mxl_rel_path?: string;
  mxl?: string;
  rating?: string | number;
};

type NormalizedEntry = CorpusEntry & {
  composerNorm: string;
  titleNorm: string;
};

type CorpusSearchProps = {
  csvUrl: string;
  onSelect: (entry: CorpusEntry) => void;
};

export default function CorpusSearch({ csvUrl, onSelect }: CorpusSearchProps) {
  const [query, setQuery] = useState("");
  const [entries, setEntries] = useState<NormalizedEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let active = true;
    setLoading(true);
    setError(null);

    fetch(csvUrl)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load ${csvUrl}`);
        }
        return res.text();
      })
      .then((text) => {
        if (!active) return;
        const parsed = Papa.parse<CorpusEntry>(text, {
          header: true,
          skipEmptyLines: true,
        });
        if (parsed.errors.length) {
          throw new Error(parsed.errors[0]?.message ?? "Failed to parse CSV");
        }
        const normalized = parsed.data.map((entry: CorpusEntry) => {
          const composer = entry.composer_name || entry.composer_label || "";
          const title = entry.title || entry.song_name || "";
          return {
            ...entry,
            composerNorm: normalizeText(composer),
            titleNorm: normalizeText(title),
          };
        });
        setEntries(normalized);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to load corpus");
      })
      .finally(() => {
        if (active) {
          setLoading(false);
        }
      });

    return () => {
      active = false;
    };
  }, [csvUrl]);

  const results = useMemo(() => {
    const pattern = normalizeQuery(query);
    if (!pattern) return [] as NormalizedEntry[];
    const regex = new RegExp(pattern);
    return entries
      .filter((entry) => regex.test(entry.titleNorm) || regex.test(entry.composerNorm))
      .sort((a, b) => parseRating(b.rating) - parseRating(a.rating))
      .slice(0, 12);
  }, [entries, query]);

  return (
    <div className="rounded-2xl border border-white/10 bg-white/70 p-4 backdrop-blur-xl dark:bg-white/5">
      <div className="flex flex-col gap-2">
        <label className="text-xs font-semibold uppercase tracking-[0.2em] text-zinc-400">Search corpus</label>
        <input
          value={query}
          onChange={(event) => setQuery(event.target.value)}
          placeholder="Search by composer or title"
          className="h-10 rounded-xl border border-white/10 bg-black/40 px-3 text-sm text-white placeholder:text-zinc-500 focus:border-blue-400 focus:outline-none"
        />
      </div>

      {loading ? (
        <div className="mt-4 text-xs text-zinc-400">Loading corpus…</div>
      ) : error ? (
        <div className="mt-4 text-xs text-red-300">{error}</div>
      ) : (
        <div className="mt-3 max-h-72 space-y-2 overflow-y-auto pr-1">
          {results.length === 0 && query.trim() ? (
            <div className="text-xs text-zinc-400">No matches.</div>
          ) : null}
          {results.map((entry, index) => {
            const composer = entry.composer_name || entry.composer_label || "Unknown";
            const title = entry.title || entry.song_name || "Untitled";
            return (
              <button
                key={`${composer}-${title}-${index}`}
                onClick={() => onSelect(entry)}
                className="w-full rounded-xl border border-white/5 bg-black/30 px-3 py-2 text-left text-xs text-zinc-100 transition hover:border-white/20 hover:bg-black/40"
              >
                <div className="text-[11px] uppercase tracking-[0.2em] text-zinc-500">{composer}</div>
                <div className="text-sm font-medium text-white">{title}</div>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}

function normalizeText(value: string): string {
  const normalized = value
    .normalize("NFKD")
    .replace(/\p{Diacritic}/gu, "")
    .replace(/[^A-Za-z0-9]+/g, " ")
    .toLowerCase();
  return normalized.trim().replace(/\s+/g, " ");
}

function normalizeQuery(query: string): string {
  const trimmed = query.trim();
  if (!trimmed) return "";
  const parts = trimmed
    .split(/\s+/)
    .map((part) => normalizeText(part))
    .filter(Boolean)
    .map((part) => part.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  if (!parts.length) return "";
  return parts.join(".*");
}

function parseRating(value: string | number | undefined): number {
  if (value === undefined || value === null) return Number.NEGATIVE_INFINITY;
  const numeric = typeof value === "number" ? value : Number.parseFloat(String(value));
  return Number.isFinite(numeric) ? numeric : Number.NEGATIVE_INFINITY;
}
