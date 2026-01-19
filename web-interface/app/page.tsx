'use client'

import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { PieceSelector } from '@/components/PieceSelector'
import { MetricsDisplay } from '@/components/MetricsDisplay'
import { PCAVisualization } from '@/components/PCAVisualization'
import { Button } from '@/components/ui/button'
import { useAppStore } from '@/lib/store'

export default function Home() {
  const { 
    setCorpusEntries, 
    setPcaData, 
    selectedPiece, 
    setPieceMetrics,
    setIsLoading,
    isLoading 
  } = useAppStore()
  
  const [initialized, setInitialized] = useState(false)

  useEffect(() => {
    async function loadInitialData() {
      try {
        // Load corpus data
        const corpusRes = await fetch('/api/corpus')
        if (corpusRes.ok) {
          const { corpusEntries } = await corpusRes.json()
          setCorpusEntries(corpusEntries || [])
        }

        // Load PCA data
        const pcaRes = await fetch('/api/pca')
        if (pcaRes.ok) {
          const { pcaData } = await pcaRes.json()
          setPcaData(pcaData || [])
        }

        setInitialized(true)
      } catch (error) {
        console.error('Error loading initial data:', error)
        setInitialized(true)
      }
    }

    loadInitialData()
  }, [setCorpusEntries, setPcaData])

  const handleAnalyzePiece = async () => {
    if (!selectedPiece) return

    setIsLoading(true)
    try {
      const response = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mxl_path: selectedPiece.mxl_path }),
      })

      if (response.ok) {
        const metrics = await response.json()
        setPieceMetrics(metrics)
      } else {
        console.error('Failed to analyze piece')
      }
    } catch (error) {
      console.error('Error analyzing piece:', error)
    } finally {
      setIsLoading(false)
    }
  }

  if (!initialized) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-slate-900 dark:border-slate-50 mx-auto mb-4"></div>
          <p className="text-slate-600 dark:text-slate-400">Loading application...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-950">
      <motion.header 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="border-b border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm"
      >
        <div className="container mx-auto px-4 py-6">
          <h1 className="text-3xl font-bold text-slate-900 dark:text-slate-50">
            JuFo 2026 - Musical Analysis Explorer
          </h1>
          <p className="text-slate-600 dark:text-slate-400 mt-2">
            Interactive visualization and analysis of musical pieces in feature space
          </p>
        </div>
      </motion.header>

      <main className="container mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left column - Piece Selection */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="lg:col-span-1"
          >
            <PieceSelector />
            
            {selectedPiece && (
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="mt-4"
              >
                <Button 
                  onClick={handleAnalyzePiece}
                  disabled={isLoading}
                  className="w-full"
                  size="lg"
                >
                  {isLoading ? 'Analyzing...' : 'Analyze Piece'}
                </Button>
              </motion.div>
            )}
          </motion.div>

          {/* Right column - Visualization and Metrics */}
          <div className="lg:col-span-2 space-y-6">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <PCAVisualization />
            </motion.div>

            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3 }}
            >
              <MetricsDisplay />
            </motion.div>
          </div>
        </div>
      </main>

      <footer className="border-t border-slate-200 dark:border-slate-800 bg-white/80 dark:bg-slate-950/80 backdrop-blur-sm mt-16">
        <div className="container mx-auto px-4 py-6 text-center text-slate-600 dark:text-slate-400">
          <p>JuFo 2026 - Computational Musicology & Stylistic Embeddings</p>
          <p className="text-sm mt-2">By Victor Gurbani</p>
        </div>
      </footer>
    </div>
  )
}
