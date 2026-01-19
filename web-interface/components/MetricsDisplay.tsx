'use client'

import React from 'react'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useAppStore } from '@/lib/store'

export function MetricsDisplay() {
  const { selectedPiece, pieceMetrics, isLoading } = useAppStore()

  if (!selectedPiece) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Piece Metrics</CardTitle>
          <CardDescription>
            Select a piece to view its computed metrics
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-slate-500">
            No piece selected
          </div>
        </CardContent>
      </Card>
    )
  }

  if (isLoading) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Piece Metrics</CardTitle>
          <CardDescription>
            Computing metrics for {selectedPiece.title}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-slate-900"></div>
          </div>
        </CardContent>
      </Card>
    )
  }

  if (!pieceMetrics) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>Piece Metrics</CardTitle>
          <CardDescription>
            {selectedPiece.title} by {selectedPiece.composer}
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-slate-500">
            Click &quot;Analyze Piece&quot; to compute metrics
          </div>
        </CardContent>
      </Card>
    )
  }

  const categories = {
    harmonic: Object.entries(pieceMetrics.harmonic || {}),
    melodic: Object.entries(pieceMetrics.melodic || {}),
    rhythmic: Object.entries(pieceMetrics.rhythmic || {})
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Piece Metrics</CardTitle>
        <CardDescription>
          {selectedPiece.title} by {selectedPiece.composer}
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div className="space-y-6">
          {Object.entries(categories).map(([category, metrics], categoryIndex) => (
            metrics.length > 0 && (
              <motion.div
                key={category}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: categoryIndex * 0.1 }}
              >
                <h3 className="text-lg font-semibold capitalize mb-3 text-slate-900 dark:text-slate-50">
                  {category} Features
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {metrics.map(([key, value], index) => (
                    <motion.div
                      key={key}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: categoryIndex * 0.1 + index * 0.05 }}
                      className="flex justify-between items-center p-3 bg-slate-50 dark:bg-slate-900 rounded-lg"
                    >
                      <span className="text-sm font-medium text-slate-700 dark:text-slate-300">
                        {key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                      </span>
                      <span className="text-sm font-mono text-slate-900 dark:text-slate-50">
                        {typeof value === 'number' ? value.toFixed(3) : String(value)}
                      </span>
                    </motion.div>
                  ))}
                </div>
              </motion.div>
            )
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
