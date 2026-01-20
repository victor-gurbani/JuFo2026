import { create } from 'zustand'

interface CorpusEntry {
  composer: string
  title: string
  mxl_path: string
  rating: number | null
  instrument: string
}

interface PCAData {
  dim1: number
  dim2: number
  dim3: number
  composer_label: string
  title: string
}

interface AppState {
  // Corpus data
  corpusEntries: CorpusEntry[]
  setCorpusEntries: (entries: CorpusEntry[]) => void
  
  // Selected piece
  selectedPiece: CorpusEntry | null
  setSelectedPiece: (piece: CorpusEntry | null) => void
  
  // Metrics for selected piece
  pieceMetrics: Record<string, any> | null
  setPieceMetrics: (metrics: Record<string, any> | null) => void
  
  // PCA visualization data
  pcaData: PCAData[]
  setPcaData: (data: PCAData[]) => void
  
  // Loading states
  isLoading: boolean
  setIsLoading: (loading: boolean) => void
  
  // Search query
  searchQuery: string
  setSearchQuery: (query: string) => void
}

export const useAppStore = create<AppState>((set) => ({
  corpusEntries: [],
  setCorpusEntries: (entries) => set({ corpusEntries: entries }),
  
  selectedPiece: null,
  setSelectedPiece: (piece) => set({ selectedPiece: piece }),
  
  pieceMetrics: null,
  setPieceMetrics: (metrics) => set({ pieceMetrics: metrics }),
  
  pcaData: [],
  setPcaData: (data) => set({ pcaData: data }),
  
  isLoading: false,
  setIsLoading: (loading) => set({ isLoading: loading }),
  
  searchQuery: '',
  setSearchQuery: (query) => set({ searchQuery: query }),
}))
