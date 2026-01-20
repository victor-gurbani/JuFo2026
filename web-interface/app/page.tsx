"use client";

import { useEffect, useMemo, useState, useRef } from "react";
import CorpusSearch, { type CorpusEntry } from "../components/CorpusSearch";
import StylisticSpace from "../components/StylisticSpace";

type AnalysisResponse = {
  ok?: boolean;
  jsonUrl?: string;
  htmlUrl?: string;
  error?: string;
};

const CANONICAL_PLOT = "/data/plots/canonical_cloud.json";
const CORPUS_CSV = "/data/stats/solo_piano_corpus.csv";
const FULL_CORPUS_CSV = "/data/stats/full_pdmx.csv";

type CarouselDef = {
  id: string;
  label: string;
  type: "html" | "image";
  items?: string[];
  folder?: string;
};

const kioskCarousels: CarouselDef[] = [
  {
    id: "highlights",
    label: "Highlights",
    type: "html",
    items: [
      "/figures/highlights/one_summers_day_pca_cloud.html",
      "/figures/highlights/ravel_string_quartet_pca_cloud.html",
    ],
  },
  {
    id: "significance",
    label: "Significance",
    type: "image",
    folder: "significance",
  },
  {
    id: "harmonic",
    label: "Harmonic",
    type: "image",
    folder: "harmonic",
  },
  {
    id: "melodic",
    label: "Melodic",
    type: "image",
    folder: "melodic",
  },
  {
    id: "rhythmic",
    label: "Rhythmic",
    type: "image",
    folder: "rhythmic",
  },
  {
    id: "annotated",
    label: "Annotated scores",
    type: "html",
    items: [
      "/figures/annotated/index.html",
    ],
  },
  {
    id: "overview",
    label: "Overview",
    type: "html",
    items: [
      "/figures/index.html",
    ],
  },
];

export default function Home() {
  const [selected, setSelected] = useState<CorpusEntry | null>(null);
  const [plotUrl, setPlotUrl] = useState(CANONICAL_PLOT);
  const [status, setStatus] = useState<string | null>(null);
  const [darkMode, setDarkMode] = useState(true);
  const [kioskMode, setKioskMode] = useState(false);
  const [filterMode, setFilterMode] = useState<"all" | "clouds" | "composers" | "highlights">("all");
  const [corpusMode, setCorpusMode] = useState<"curated" | "full">("curated");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isWide, setIsWide] = useState(true);
  const [carouselFolderIndex, setCarouselFolderIndex] = useState(0);
  const [carouselItemIndex, setCarouselItemIndex] = useState(0);
  const [carouselPhase, setCarouselPhase] = useState<"pca" | "figure">("pca");
  const [manualViewMode, setManualViewMode] = useState<"pca" | string>("pca");
  const [imageMap, setImageMap] = useState<Record<string, string[]>>({});
  const [modalImage, setModalImage] = useState<string | null>(null);
  
  const fullscreenRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    // Preload images for dynamic folders
    kioskCarousels.forEach(async (c) => {
      if (c.type === "image" && c.folder && !imageMap[c.folder]) {
        try {
          const res = await fetch(`/api/images?folder=${c.folder}`);
          const data = await res.json();
          if (data.images && Array.isArray(data.images) && data.images.length > 0) {
            setImageMap((prev) => ({ ...prev, [c.folder!]: data.images }));
          }
        } catch (e) {
          console.error("Failed to load images for", c.folder, e);
        }
      }
    });
  }, []);

  useEffect(() => {
    const root = document.documentElement;
    if (darkMode) {
      root.classList.add("dark");
    } else {
      root.classList.remove("dark");
    }
  }, [darkMode]);

  useEffect(() => {
    const handleResize = () => {
      setIsWide(window.innerWidth >= 1280);
    };
    handleResize();
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  useEffect(() => {
    const handleFullscreenChange = () => {
      const active = Boolean(document.fullscreenElement);
      setIsFullscreen(active);
      setKioskMode(active);
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  useEffect(() => {
    if (!kioskMode) return;
    const modes: Array<typeof filterMode> = ["clouds", "composers", "highlights", "all"];
    const interval = window.setInterval(() => {
      setFilterMode((current) => {
        const idx = modes.indexOf(current);
        return modes[(idx + 1) % modes.length];
      });
    }, 12000);
    return () => window.clearInterval(interval);
  }, [kioskMode]);

  useEffect(() => {
    if (!kioskMode) return;
    const interval = window.setInterval(() => {
      setCarouselFolderIndex((currentFolderIdx) => {
        const nextFolderIdx = (currentFolderIdx + 1) % kioskCarousels.length;
        const nextDef = kioskCarousels[nextFolderIdx];
        
        let count = 1;
        if (nextDef.type === "html" && nextDef.items) {
          count = nextDef.items.length;
        } else if (nextDef.type === "image" && nextDef.folder) {
          // If images, we step through them.
          // To ensure we see multiple sets of images before switching folder, 
          // we might want checking (prevItemIndex + step) vs length.
          // But here we rely on the 14s interval. 
          // If we want "sequential change", we increment item index.
          // Note: The logic here increments FOLDER every 14s. 
          // Wait, user wants sequential image change "not all at once".
          // If we change folder every 14s, we only see 14s of that folder.
          // We probably want to stay on the folder for longer if it has many images?
          // Or change the interval logic.
          
          // Let's keep consistent folder switching for now to avoid complexity explosion,
          // but update the Item Index so next time we come back to this folder (or if we calc global index), it shifts.
          // Actually, `setCarouselItemIndex` tracks a single index number. 
          // If we map that index to the current folder's items, it effectively shuffles/rotates.
          count = imageMap[nextDef.folder]?.length || 1;
        }

        setCarouselItemIndex((prev) => (prev + 1) % 1000); // Keep it growing / rotating
        return nextFolderIdx;
      });
      setCarouselPhase((current) => (current === "pca" ? "figure" : "pca"));
    }, 14000);
    return () => window.clearInterval(interval);
  }, [kioskMode, imageMap]);

  const selectedTitle = useMemo(() => {
    if (!selected) return "No selection";
    return selected.title || selected.song_name || "Untitled";
  }, [selected]);

  const selectedComposer = useMemo(() => {
    if (!selected) return "";
    return selected.composer_name || selected.composer_label || "Unknown";
  }, [selected]);

  const selectedPath = useMemo(() => {
    if (!selected) return "";
    return (
      selected.mxl_abs_path ||
      selected.mxl_path ||
      selected.mxl_rel_path ||
      selected.mxl ||
      ""
    );
  }, [selected]);

  const corpusUrl = useMemo(() => {
    return corpusMode === "full" ? FULL_CORPUS_CSV : CORPUS_CSV;
  }, [corpusMode]);

  const runAnalysis = async () => {
    if (!selectedPath) {
      setStatus("Select a piece first.");
      return;
    }
    setStatus("Analyzing piece and projecting onto PCA map…");
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mxlPath: selectedPath,
          title: selectedTitle,
          composer: selectedComposer || "External",
        }),
      });
      const payload = (await response.json()) as AnalysisResponse;
      if (!response.ok || payload.error || !payload.jsonUrl) {
        throw new Error(payload.error || "Analysis failed");
      }
      setPlotUrl(`${payload.jsonUrl}?t=${Date.now()}`);
      setStatus("Analysis complete. Plot updated.");
    } catch (error) {
      setStatus(error instanceof Error ? error.message : "Analysis failed");
    }
  };

  const currentCarousel = useMemo(() => {
     if (kioskMode) return kioskCarousels[carouselFolderIndex] ?? kioskCarousels[0];
     if (manualViewMode !== "pca") return kioskCarousels.find(c => c.id === manualViewMode) || kioskCarousels[0];
     return kioskCarousels[0];
  }, [kioskMode, carouselFolderIndex, manualViewMode]);
  
  // Resolve items for current carousel
  const currentItems = useMemo(() => {
    if (currentCarousel.type === "html") {
      return currentCarousel.items || [];
    }
    if (currentCarousel.type === "image" && currentCarousel.folder) {
      return imageMap[currentCarousel.folder] || [];
    }
    return [];
  }, [currentCarousel, imageMap]);
  
  // For HTML, single item. For Images, slice of 3.
  // We use carouselItemIndex to offset.
  const activeItems = useMemo(() => {
    if (currentItems.length === 0) return [];
    
    // In Manual Mode (and not Kiosk), if it's images, we want to show ALL of them (or many),
    // not just a slice. Or at least a larger grid.
    if (!kioskMode && currentCarousel.type === "image") {
        return currentItems; 
    }
    
    if (currentCarousel.type === "html") {
      // In manual mode for HTML, maybe cycle? or just show first?
      // Let's just show first for now, or all if we can stack them?
      // HTMLs are usually heavy, so just first is safer.
      // Or we can add sub-navigation later.
      return [currentItems[0]]; // Always first in manual for stability
    }
    
    // Kiosk Mode (Images) -> Sliding Window
    const start = carouselItemIndex % currentItems.length;
    const result = [];
    for (let i = 0; i < 3; i++) {
        result.push(currentItems[(start + i) % currentItems.length]);
    }
    return result;
  }, [currentItems, carouselItemIndex, currentCarousel.type, kioskMode]);

  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        if (fullscreenRef.current) {
            await fullscreenRef.current.requestFullscreen();
        } else {
            await document.documentElement.requestFullscreen();
        }
      } else {
        await document.exitFullscreen();
      }
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div className="min-h-screen bg-zinc-50 text-zinc-900 transition-colors dark:bg-[#04060c] dark:text-white">
      <div className="mx-auto flex min-h-screen max-w-[90%] flex-col gap-10 px-6 py-10 lg:flex-row">
        <section className="flex w-full flex-col gap-6 lg:max-w-sm">
          <div className="rounded-2xl border border-white/10 bg-white/70 p-6 shadow-xl shadow-blue-500/10 backdrop-blur-xl dark:bg-white/5">
            <p className="text-xs font-semibold uppercase tracking-[0.3em] text-blue-500 dark:text-blue-200">Computational Narrative</p>
            <h1 className="mt-4 text-3xl font-semibold leading-tight">
              Empirische Musikalische Kartographie
            </h1>
            <p className="mt-1 text-lg font-medium text-zinc-500 dark:text-zinc-400">Victor Gurbani</p>
            <p className="mt-3 text-sm text-zinc-600 dark:text-zinc-300">
              Explore stylistic evolution in real time. Load precomputed PCA clouds or analyze a new score
              from the corpus to project it into the stylistic space.
            </p>
          </div>

          <div className="rounded-2xl border border-white/10 bg-white/70 p-4 backdrop-blur-xl dark:bg-white/5">
            <div className="text-xs uppercase tracking-[0.2em] text-zinc-500 dark:text-zinc-400">Display modes</div>
            <div className="mt-3 grid grid-cols-2 gap-2 text-xs">
              <button
                onClick={() => setKioskMode((value) => !value)}
                className={`h-9 rounded-xl border px-3 transition ${
                  kioskMode
                    ? "border-blue-400 bg-blue-500/20 text-blue-600 dark:text-blue-200"
                    : "border-white/20 bg-white/40 text-zinc-700 hover:border-blue-300 dark:bg-black/30 dark:text-zinc-200"
                }`}
              >
                {kioskMode ? "Kiosk: on" : "Kiosk: off"}
              </button>
              <button
                onClick={() => setDarkMode((value) => !value)}
                className={`h-9 rounded-xl border px-3 transition ${
                  darkMode
                    ? "border-blue-400 bg-blue-500/20 text-blue-600 dark:text-blue-200"
                    : "border-white/20 bg-white/40 text-zinc-700 hover:border-blue-300 dark:bg-black/30 dark:text-zinc-200"
                }`}
              >
                {darkMode ? "Dark mode" : "Light mode"}
              </button>
              <button
                onClick={toggleFullscreen}
                className={`col-span-2 h-9 rounded-xl border px-3 transition ${
                  isFullscreen
                    ? "border-blue-400 bg-blue-500/20 text-blue-600 dark:text-blue-200"
                    : "border-white/20 bg-white/40 text-zinc-700 hover:border-blue-300 dark:bg-black/30 dark:text-zinc-200"
                }`}
              >
                {isFullscreen ? "Exit fullscreen" : "Fullscreen (auto kiosk)"}
              </button>
               <select
                value={manualViewMode}
                onChange={(e) => setManualViewMode(e.target.value)}
                disabled={kioskMode}
                className={`col-span-2 h-9 rounded-xl border px-3 transition appearance-none bg-transparent ${
                  manualViewMode !== "pca"
                    ? "border-blue-400 bg-blue-500/20 text-blue-600 dark:text-blue-200"
                    : "border-white/20 bg-white/40 text-zinc-700 hover:border-blue-300 dark:bg-black/30 dark:text-zinc-200"
                } disabled:opacity-50`}
              >
                <option value="pca" className="bg-white dark:bg-black">Interactive Stylistic Space</option>
                {kioskCarousels.map((c) => (
                    <option key={c.id} value={c.id} className="bg-white dark:bg-black">View: {c.label}</option>
                ))}
              </select>
              <button
                onClick={() => setCorpusMode(corpusMode === "curated" ? "full" : "curated")}
                className={`col-span-2 h-9 rounded-xl border px-3 transition ${
                  corpusMode === "full"
                    ? "border-blue-400 bg-blue-500/20 text-blue-600 dark:text-blue-200"
                    : "border-white/20 bg-white/40 text-zinc-700 hover:border-blue-300 dark:bg-black/30 dark:text-zinc-200"
                }`}
              >
                {corpusMode === "full" ? "Full corpus (PDMX)" : "Curated corpus"}
              </button>
            </div>
          </div>

          <CorpusSearch
            key={corpusUrl}
            csvUrl={corpusUrl}
            onSelect={(entry) => {
              setSelected(entry);
              setStatus(null);
            }}
          />

          <div className="rounded-2xl border border-white/10 bg-white/70 p-4 backdrop-blur-xl dark:bg-white/5">
            <div className="text-xs uppercase tracking-[0.2em] text-zinc-500 dark:text-zinc-400">Selected piece</div>
            <div className="mt-2 text-lg font-semibold text-zinc-900 dark:text-white">{selectedTitle}</div>
            <div className="text-sm text-zinc-600 dark:text-zinc-400">{selectedComposer || ""}</div>
            <div className="mt-4 flex flex-col gap-2">
              <button
                onClick={runAnalysis}
                className="h-10 rounded-xl bg-blue-500/90 text-sm font-semibold text-white transition hover:bg-blue-400"
              >
                Analyze + project
              </button>
              <button
                onClick={() => {
                  setPlotUrl(`${CANONICAL_PLOT}?t=${Date.now()}`);
                  setFilterMode("all");
                  setStatus("Showing canonical PCA cloud.");
                }}
                className="h-10 rounded-xl border border-white/20 text-sm text-zinc-700 transition hover:border-blue-300 dark:border-white/10 dark:text-zinc-200"
              >
                Reset to canonical
              </button>
              {status ? <div className="text-xs text-zinc-500 dark:text-zinc-400">{status}</div> : null}
            </div>
          </div>
        </section>

        <section className="flex min-h-[480px] w-full flex-1 flex-col gap-4">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-xs uppercase tracking-[0.25em] text-zinc-500 dark:text-zinc-500">Stylistic space</p>
              <h2 className="text-xl font-semibold">Composer PCA Cloud</h2>
            </div>
            <div className="flex flex-wrap gap-2 text-xs">
              {[
                { id: "all", label: "All traces" },
                { id: "clouds", label: "PCA clouds" },
                { id: "composers", label: "Composer points" },
                { id: "highlights", label: "Highlights" },
              ].map((option) => (
                <button
                  key={option.id}
                  onClick={() => setFilterMode(option.id as typeof filterMode)}
                  className={`h-8 rounded-full border px-3 transition ${
                    filterMode === option.id
                      ? "border-blue-400 bg-blue-500/20 text-blue-600 dark:text-blue-200"
                      : "border-white/20 bg-white/60 text-zinc-600 hover:border-blue-300 dark:bg-black/30 dark:text-zinc-300"
                  }`}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
          
          <div ref={fullscreenRef} className="relative flex h-[800px] flex-col gap-4 bg-zinc-50 p-2 text-zinc-900 transition-colors dark:bg-[#04060c] dark:text-white overflow-y-auto lg:overflow-visible rounded-3xl pb-10 lg:pb-0">
            {/* Added Wrapper with Ref and BG colors to ensure fullscreen looks correct */}
              
            {kioskMode && (
                <div className="absolute top-12 left-12 z-50 pointer-events-none drop-shadow-[0_2px_4px_rgba(0,0,0,0.5)] max-w-4xl">
                    <h1 className="text-6xl font-bold text-zinc-900 dark:text-white opacity-90 leading-tight">Empirische Musikalische Kartographie</h1>
                    <p className="mt-4 text-3xl font-medium text-zinc-700 dark:text-zinc-300 opacity-90">Victor Gurbani</p>
                </div>
            )}
              
          {kioskMode && isWide ? (
            <div className="grid h-full w-full gap-4 lg:grid-cols-2">
              <div className="rounded-3xl border border-white/10 bg-gradient-to-br from-white/80 via-white/40 to-white/60 p-4 shadow-2xl shadow-blue-500/10 backdrop-blur-2xl dark:from-[#0a1220] dark:via-[#04060c] dark:to-[#05070d]">
                <StylisticSpace dataUrl={plotUrl} className="h-full w-full" filterMode={filterMode} kioskMode={kioskMode} />
              </div>
              <div className="relative overflow-hidden rounded-3xl border border-white/10 bg-white/70 shadow-2xl shadow-blue-500/10 backdrop-blur-2xl dark:bg-white/5 p-4 flex flex-col justify-center">
                {currentCarousel.type === "html" ? (
                    <iframe
                    title={`carousel-${currentCarousel.id}`}
                    src={activeItems[0]}
                    className="h-full w-full"
                    />
                ) : (
                    <div className="grid h-full w-full grid-cols-2 grid-rows-2 gap-4">
                        {activeItems.map((src, i) => {
                             let spanClass = "col-span-1 row-span-1";
                             if (activeItems.length === 1) spanClass = "col-span-2 row-span-2";
                             else if (activeItems.length === 3 && i === 0) spanClass = "col-span-2 row-span-1";
                             
                             return (
                                 <div key={i} className={`relative flex items-center justify-center overflow-hidden rounded-xl bg-white/10 p-2 ${spanClass} cursor-pointer hover:bg-white/20 transition`} onClick={() => setModalImage(src)}>
                                    <img src={src} alt="Analysis Figure" className="max-h-full max-w-full object-contain" />
                                 </div>
                             );
                        })}
                    </div>
                )}
                
                <div className="pointer-events-none absolute bottom-4 left-4 rounded-full bg-black/60 px-3 py-1 text-xs text-white z-10">
                  {currentCarousel.label}
                </div>
              </div>
            </div>
          ) : (
            <div className="h-full w-full rounded-3xl border border-white/10 bg-gradient-to-br from-white/80 via-white/40 to-white/60 p-4 shadow-2xl shadow-blue-500/10 backdrop-blur-2xl dark:from-[#0a1220] dark:via-[#04060c] dark:to-[#05070d]">
              {(kioskMode && carouselPhase === "figure") || (!kioskMode && manualViewMode !== "pca") ? (
                 <div className="relative h-full w-full rounded-2xl overflow-y-auto">
                    {currentCarousel.type === "html" ? (
                        <iframe
                            title={`carousel-${currentCarousel.id}`}
                            src={activeItems[0]}
                            className="h-full w-full min-h-[500px]"
                        />
                    ) : (
                         <div className={`grid h-full w-full gap-4 ${!kioskMode ? 'grid-cols-1 md:grid-cols-2 lg:grid-cols-3 auto-rows-min' : 'grid-cols-2 grid-rows-2'}`}>
                            {activeItems.map((src, i) => {
                                let spanClass = "";
                                if (kioskMode) {
                                     spanClass = "col-span-1 row-span-1";
                                     if (activeItems.length === 1) spanClass = "col-span-2 row-span-2";
                                     else if (activeItems.length === 3 && i === 0) spanClass = "col-span-2 row-span-1";
                                }
                                
                                return (
                                <div key={i} className={`relative flex items-center justify-center overflow-hidden rounded-xl bg-white/10 p-2 min-h-[250px] ${spanClass} cursor-pointer hover:bg-white/20 transition`} onClick={() => setModalImage(src)}>
                                    <img src={src} alt="Analysis Figure" className="max-h-full max-w-full object-contain" />
                                </div>
                            )})}
                        </div>
                    )}
                    <div className="pointer-events-none sticky bottom-4 left-4 inline-block rounded-full bg-black/60 px-3 py-1 text-xs text-white z-10 mx-4 mb-4">
                        {currentCarousel.label}
                    </div>
                 </div>
              ) : (
                <StylisticSpace dataUrl={plotUrl} className="h-full w-full" filterMode={filterMode} kioskMode={kioskMode} />
              )}
            </div>
          )}
          
            {/* Image Modal */}
            {modalImage && (
                <div 
                    className="absolute inset-0 z-[100] flex items-center justify-center bg-black/80 backdrop-blur-sm p-10 cursor-pointer"
                    onClick={() => setModalImage(null)}
                >
                    <img 
                        src={modalImage} 
                        alt="Zoomed Figure" 
                        className="max-h-full max-w-full object-contain drop-shadow-2xl rounded-lg"
                    />
                    <button className="absolute top-4 right-4 rounded-full bg-white/10 p-2 text-white hover:bg-white/20">
                        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
                    </button>
                </div>
            )}
            
          </div>
          
          <p className="text-xs text-zinc-500 dark:text-zinc-400">
            This view loads Plotly JSON files exported by the Python pipeline. The canonical cloud is
            static; analysis calls compute new projections live and overwrite the temporary JSON.
          </p>
        </section>
      </div>
    </div>
  );
}
