"use client";

import { type ComponentType, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

import { loadComposerYearMap, resolveComposerYear, type ComposerYearMap } from "../lib/composerYears";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false }) as ComponentType<any>;

type PlotlyFigure = {
  data: unknown[];
  layout: Record<string, unknown>;
};

type FilterMode = "all" | "clouds" | "composers" | "highlights";

type StylisticSpaceProps = {
  dataUrl: string;
  className?: string;
  onPointClick?: (payload: unknown) => void;
  filterMode?: FilterMode;
  kioskMode?: boolean;
};

function toNumberArray(value: unknown): number[] | null {
  if (Array.isArray(value)) {
    return value.map((v) => Number(v)).filter((v) => Number.isFinite(v));
  }

  if (value && typeof value === "object") {
    const maybe = value as { dtype?: unknown; bdata?: unknown };
    const dtype = typeof maybe.dtype === "string" ? maybe.dtype : null;
    const bdata = typeof maybe.bdata === "string" ? maybe.bdata : null;
    if (!dtype || !bdata) return null;

    if (typeof globalThis.atob !== "function") return null;

    // Plotly JSON can encode numeric arrays as base64 binary blobs (bdata) with a numpy dtype.
    // Example: {"dtype":"f8","bdata":"..."}
    const bin = globalThis.atob(bdata);
    const bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) {
      bytes[i] = bin.charCodeAt(i);
    }

    const byteLength = bytes.byteLength;
    const byteOffset = bytes.byteOffset;
    const buffer = bytes.buffer;

    const build = <T extends ArrayBufferView>(
      ctor: { new (buffer: ArrayBuffer, byteOffset: number, length: number): T },
      bytesPerElement: number,
    ) => {
      if (byteLength % bytesPerElement !== 0) return null;
      const length = byteLength / bytesPerElement;
      try {
        const view = new ctor(buffer, byteOffset, length) as unknown as ArrayLike<number>;
        return Array.from(view).map((v) => Number(v)).filter((v) => Number.isFinite(v));
      } catch {
        return null;
      }
    };

    switch (dtype) {
      case "f8":
        return build(Float64Array, 8);
      case "f4":
        return build(Float32Array, 4);
      case "i4":
        return build(Int32Array, 4);
      case "i2":
        return build(Int16Array, 2);
      case "u1":
        return build(Uint8Array, 1);
      default:
        return null;
    }
  }

  return null;
}

export default function StylisticSpace({
  dataUrl,
  className,
  onPointClick,
  filterMode = "all",
  kioskMode = false,
}: StylisticSpaceProps) {
  const [figure, setFigure] = useState<PlotlyFigure | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [cameraAngle, setCameraAngle] = useState(0);
  const [figureRevision, setFigureRevision] = useState(0);

  const [composerYears, setComposerYears] = useState<ComposerYearMap | null>(null);

  const [showAgeTrajectory, setShowAgeTrajectory] = useState(true);
  const [useBezierTrajectory, setUseBezierTrajectory] = useState(false);
  const [hideCloudSurfaces, setHideCloudSurfaces] = useState(false);

  useEffect(() => {
    let active = true;
    setError(null);
    setFigure(null);

    // Force Plotly to refresh when a new figure arrives.
    setFigureRevision((v) => v + 1);

    fetch(dataUrl)
      .then((res) => {
        if (!res.ok) {
          throw new Error(`Failed to load ${dataUrl}`);
        }
        return res.json();
      })
      .then((payload) => {
        if (!active) return;
        setFigure(payload);
        setFigureRevision((v) => v + 1);
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to load plot");
      });

    return () => {
      active = false;
    };
  }, [dataUrl]);

  useEffect(() => {
    let active = true;
    loadComposerYearMap()
      .then((map) => {
        if (!active) return;
        setComposerYears(map);
      })
      .catch(() => {
        // If years can't be loaded, we still allow drawing a line in fallback order.
        if (!active) return;
        setComposerYears(null);
      });
    return () => {
      active = false;
    };
  }, []);

  const filteredData = useMemo(() => {
    if (!figure?.data) return [];
    return (figure.data as Array<Record<string, unknown>>).filter((trace) => {
      if (filterMode === "all") return true;
      if (filterMode === "clouds") return trace.type === "isosurface";
      if (filterMode === "composers") {
        return trace.type === "scatter3d" && trace.name?.toString().includes("pieces");
      }
      if (filterMode === "highlights") {
        return trace.type === "scatter3d" && trace.marker && (trace.marker as Record<string, unknown>).symbol === "diamond";
      }
      return true;
    });
  }, [figure, filterMode]);

  type Centroid = { composer: string; year: number | null; x: number; y: number; z: number };

  const centroids = useMemo((): Centroid[] => {
    if (!figure?.data) return [];
    const out: Centroid[] = [];

    for (const raw of figure.data as Array<Record<string, unknown>>) {
      if (raw.type !== "scatter3d") continue;

      const name = raw.name?.toString() ?? "";
      if (!name) continue;

      const xs = toNumberArray(raw.x);
      const ys = toNumberArray(raw.y);
      const zs = toNumberArray(raw.z);
      if (!xs || !ys || !zs) continue;
      if (xs.length < 1 || ys.length < 1 || zs.length < 1) continue;

      const mean = (vals: number[]) => vals.reduce((a, b) => a + b, 0) / vals.length;
      out.push({ composer: name, year: resolveComposerYear(composerYears, name), x: mean(xs), y: mean(ys), z: mean(zs) });
    }

    return out;
  }, [figure, composerYears]);

  const trajectoryTraces = useMemo(() => {
    if (!showAgeTrajectory) return [] as Array<Record<string, unknown>>;
    if (centroids.length < 2) return [] as Array<Record<string, unknown>>;

    const known = centroids.filter((c) => typeof c.year === "number" && Number.isFinite(c.year));
    const ordered = (known.length >= 2 ? known : centroids)
      .slice()
      .sort((a, b) => {
        if (a.year != null && b.year != null) return a.year - b.year;
        if (a.year != null) return -1;
        if (b.year != null) return 1;
        return a.composer.localeCompare(b.composer);
      });

    if (ordered.length < 2) return [] as Array<Record<string, unknown>>;

    const pts = ordered.map((c) => ({ x: c.x, y: c.y, z: c.z }));

    const bezierSample = (p0: any, p1: any, p2: any, p3: any, steps: number) => {
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const u = 1 - t;
        const b0 = u * u * u;
        const b1 = 3 * u * u * t;
        const b2 = 3 * u * t * t;
        const b3 = t * t * t;
        xs.push(b0 * p0.x + b1 * p1.x + b2 * p2.x + b3 * p3.x);
        ys.push(b0 * p0.y + b1 * p1.y + b2 * p2.y + b3 * p3.y);
        zs.push(b0 * p0.z + b1 * p1.z + b2 * p2.z + b3 * p3.z);
      }
      return { xs, ys, zs };
    };

    const buildSmoothCurve = (points: Array<{ x: number; y: number; z: number }>) => {
      // Catmull-Rom -> cubic Bezier per segment, sampled for Plotly.
      const xs: number[] = [];
      const ys: number[] = [];
      const zs: number[] = [];
      const steps = 18;
      for (let i = 0; i < points.length - 1; i++) {
        const p0 = points[Math.max(0, i - 1)];
        const p1 = points[i];
        const p2 = points[i + 1];
        const p3 = points[Math.min(points.length - 1, i + 2)];

        const c1 = { x: p1.x + (p2.x - p0.x) / 6, y: p1.y + (p2.y - p0.y) / 6, z: p1.z + (p2.z - p0.z) / 6 };
        const c2 = { x: p2.x - (p3.x - p1.x) / 6, y: p2.y - (p3.y - p1.y) / 6, z: p2.z - (p3.z - p1.z) / 6 };

        const seg = bezierSample(p1, c1, c2, p2, steps);
        // Avoid duplicating the first point of each segment.
        const start = i === 0 ? 0 : 1;
        xs.push(...seg.xs.slice(start));
        ys.push(...seg.ys.slice(start));
        zs.push(...seg.zs.slice(start));
      }
      return { xs, ys, zs };
    };

    const line = useBezierTrajectory ? buildSmoothCurve(pts) : {
      xs: pts.map((p) => p.x),
      ys: pts.map((p) => p.y),
      zs: pts.map((p) => p.z),
    };

    const markerText = ordered.map((c) => (c.year != null ? `${c.composer} (${c.year})` : c.composer));

    return [
      {
        type: "scatter3d",
        mode: "lines",
        x: line.xs,
        y: line.ys,
        z: line.zs,
        opacity: 1,
        line: { color: "rgba(255,210,0,1)", width: 10 },
        hoverinfo: "skip",
        showlegend: false,
        name: "age-trajectory",
      },
      {
        type: "scatter3d",
        mode: "markers",
        x: pts.map((p) => p.x),
        y: pts.map((p) => p.y),
        z: pts.map((p) => p.z),
        marker: { size: 6, color: "rgba(255,210,0,1)", opacity: 0.95 },
        text: markerText,
        hovertemplate: "%{text}<extra></extra>",
        showlegend: false,
        name: "age-centroids",
      },
    ] as Array<Record<string, unknown>>;
  }, [centroids, showAgeTrajectory, useBezierTrajectory]);

  const plotRevision = useMemo(() => {
    const bits = (showAgeTrajectory ? 1 : 0) + (useBezierTrajectory ? 2 : 0) + (hideCloudSurfaces ? 4 : 0);
    return figureRevision * 10 + bits;
  }, [figureRevision, showAgeTrajectory, useBezierTrajectory, hideCloudSurfaces]);

  const displayData = useMemo(() => {
    const base = filteredData as Array<Record<string, unknown>>;
    const out: Array<Record<string, unknown>> = [];

    for (const trace of base) {
      if (hideCloudSurfaces && trace.type === "isosurface") continue;

      if (hideCloudSurfaces && trace.type === "scatter3d") {
        const mode = trace.mode?.toString() ?? "";
        if (mode.includes("markers")) {
          const marker = (trace.marker as Record<string, unknown> | undefined) ?? {};
          out.push({
            ...trace,
            marker: {
              ...marker,
              opacity: 0.14,
              size: typeof marker.size === "number" ? Math.min(marker.size, 2) : 2,
            },
          });
          continue;
        }
      }

      out.push(trace);
    }

    if (showAgeTrajectory) {
      out.push(...trajectoryTraces);
    }

    return out;
  }, [filteredData, hideCloudSurfaces, showAgeTrajectory, trajectoryTraces]);

  const layout = useMemo(() => {
    if (!figure?.layout) return null;
    const incoming = figure.layout as Record<string, unknown>;
    const scene = (incoming.scene as Record<string, unknown>) ?? {};

    const axisTemplate = {
      showgrid: true,
      gridcolor: "rgba(148,163,184,0.25)",
      gridwidth: 1,
      zeroline: false,
      showticklabels: false,
      ticks: "",
      showspikes: false,
    };

    const preferredTitles = [
      "PC1 (Chromatik/Dissonanz)",
      "PC2 (Dichte/Klarheit)",
      "PC3 (Registral/Textur)",
    ];

    const normalizeAxis = (
      axis: Record<string, unknown> | undefined,
      fallback: string,
    ) => {
      const currentTitle =
        (axis?.title as { text?: string } | undefined)?.text ?? (axis?.title as string | undefined);
      const needsOverride = !currentTitle || /^dim[123]$/i.test(currentTitle.trim());
      const title = needsOverride ? { text: fallback, font: { color: "#94a3b8" } } : axis?.title;
      return {
        ...axisTemplate,
        ...(axis ?? {}),
        title,
      };
    };

    const base = {
      ...incoming,
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      margin: { l: 0, r: 0, t: 0, b: 0 },
      showlegend: false,
      scene: {
        ...scene,
        aspectmode: "cube",
        xaxis: normalizeAxis(scene.xaxis as Record<string, unknown> | undefined, preferredTitles[0]),
        yaxis: normalizeAxis(scene.yaxis as Record<string, unknown> | undefined, preferredTitles[1]),
        zaxis: normalizeAxis(scene.zaxis as Record<string, unknown> | undefined, preferredTitles[2]),
      },
    };
    if (kioskMode) {
      const radius = 2.1;
      return {
        ...base,
        scene: {
          ...base.scene,
          camera: {
            eye: {
              x: radius * Math.cos(cameraAngle),
              y: radius * Math.sin(cameraAngle),
              z: 1.2,
            },
          },
        },
      };
    }
    return base;
  }, [cameraAngle, figure, kioskMode]);

  useEffect(() => {
    if (!kioskMode) return;
    let frame = 0;
    let lastTime = performance.now();
    const tick = (now: number) => {
      const delta = now - lastTime;
      lastTime = now;
      const speed = 0.00025;
      setCameraAngle((value) => value + delta * speed);
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [kioskMode]);

  if (error) {
    return (
      <div className={`flex h-full items-center justify-center rounded-3xl border border-white/10 bg-white/5 p-8 text-sm text-zinc-300 ${className ?? ""}`}>
        {error}
      </div>
    );
  }

  if (!figure || !layout) {
    return (
      <div className={`flex h-full items-center justify-center rounded-3xl border border-white/10 bg-white/5 p-8 text-sm text-zinc-300 ${className ?? ""}`}>
        Loading stylistic space…
      </div>
    );
  }

  return (
    <div className={`relative h-full w-full ${className ?? ""}`}>
      {!kioskMode ? (
        <div className="pointer-events-auto absolute left-3 top-3 z-10 rounded-2xl border border-white/10 bg-black/50 p-3 text-xs text-zinc-200 backdrop-blur">
          <label className="flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={showAgeTrajectory}
              onChange={(e) => setShowAgeTrajectory(e.target.checked)}
            />
            <span>Connect composers by age</span>
          </label>
          <label className={`mt-2 flex cursor-pointer items-center gap-2 ${!showAgeTrajectory ? "opacity-50" : ""}`}>
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={useBezierTrajectory}
              onChange={(e) => setUseBezierTrajectory(e.target.checked)}
              disabled={!showAgeTrajectory}
            />
            <span>Bezier curve</span>
          </label>
          <label className="mt-2 flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              className="h-4 w-4"
              checked={hideCloudSurfaces}
              onChange={(e) => setHideCloudSurfaces(e.target.checked)}
            />
            <span>Hide clouds (dots + line only)</span>
          </label>
        </div>
      ) : null}
      <Plot
        data={displayData}
        layout={layout}
        revision={plotRevision}
        config={{ displayModeBar: false, responsive: true, scrollZoom: true }}
        className="h-full w-full"
        onClick={(event: unknown) => onPointClick?.(event)}
      />
      <div className="pointer-events-none absolute inset-0 rounded-3xl border border-white/10" />
    </div>
  );
}
