"use client";

import { type ComponentType, useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";

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

  useEffect(() => {
    let active = true;
    setError(null);
    setFigure(null);

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
      })
      .catch((err) => {
        if (!active) return;
        setError(err instanceof Error ? err.message : "Failed to load plot");
      });

    return () => {
      active = false;
    };
  }, [dataUrl]);

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

  const layout = useMemo(() => {
    if (!figure?.layout) return null;
    const incoming = figure.layout as Record<string, unknown>;
    const scene = (incoming.scene as Record<string, unknown>) ?? {};
    const axisTemplate = {
      showgrid: false,
      zeroline: false,
      showticklabels: false,
      // title: "", // Let the plot layout determine the title (or use default from file)
      ticks: "",
      showspikes: false,
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
        xaxis: { ...axisTemplate, ...(scene.xaxis as Record<string, unknown>) },
        yaxis: { ...axisTemplate, ...(scene.yaxis as Record<string, unknown>) },
        zaxis: { ...axisTemplate, ...(scene.zaxis as Record<string, unknown>) },
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
      <Plot
        data={filteredData}
        layout={layout}
        config={{ displayModeBar: false, responsive: true, scrollZoom: true }}
        className="h-full w-full"
        onClick={(event: unknown) => onPointClick?.(event)}
      />
      <div className="pointer-events-none absolute inset-0 rounded-3xl border border-white/10" />
    </div>
  );
}
