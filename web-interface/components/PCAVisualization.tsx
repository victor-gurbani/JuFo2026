'use client'

import React, { useEffect, useState } from 'react'
import dynamic from 'next/dynamic'
import { motion } from 'framer-motion'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useAppStore } from '@/lib/store'

// Dynamically import Plotly to avoid SSR issues
const Plot = dynamic(() => import('react-plotly.js'), { ssr: false })

export function PCAVisualization() {
  const { pcaData, selectedPiece, pieceMetrics } = useAppStore()
  const [plotData, setPlotData] = useState<any[]>([])
  const [layout, setLayout] = useState<any>({})

  useEffect(() => {
    if (!pcaData || pcaData.length === 0) {
      return
    }

    // Group data by composer
    const composerGroups: Record<string, typeof pcaData> = {}
    pcaData.forEach(point => {
      if (!composerGroups[point.composer_label]) {
        composerGroups[point.composer_label] = []
      }
      composerGroups[point.composer_label].push(point)
    })

    // Create traces for each composer
    const traces = Object.entries(composerGroups).map(([composer, points]) => ({
      x: points.map(p => p.dim1),
      y: points.map(p => p.dim2),
      z: points.map(p => p.dim3),
      mode: 'markers',
      type: 'scatter3d',
      name: composer,
      text: points.map(p => `${p.title}<br>${p.composer_label}`),
      hovertemplate: '<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>',
      marker: {
        size: 4,
        opacity: 0.7,
      },
    }))

    // If there's a selected piece with metrics, add it as a highlighted point
    if (selectedPiece && pieceMetrics && pieceMetrics.pca) {
      traces.push({
        x: [pieceMetrics.pca.dim1],
        y: [pieceMetrics.pca.dim2],
        z: [pieceMetrics.pca.dim3],
        mode: 'markers',
        type: 'scatter3d',
        name: 'Selected Piece',
        text: [`${selectedPiece.title}<br>${selectedPiece.composer}`],
        hovertemplate: '<b>%{text}</b><br>PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>',
        marker: {
          size: 12,
          color: '#ff0000' as any,
          symbol: 'diamond' as any,
          line: {
            color: '#ffffff',
            width: 2
          } as any
        } as any,
      })
    }

    setPlotData(traces)

    setLayout({
      title: 'PCA Feature Space',
      autosize: true,
      scene: {
        xaxis: { title: 'PC1' },
        yaxis: { title: 'PC2' },
        zaxis: { title: 'PC3' },
        camera: {
          eye: { x: 1.5, y: 1.5, z: 1.5 }
        }
      },
      margin: { l: 0, r: 0, t: 40, b: 0 },
      legend: {
        x: 1.02,
        y: 1,
        xanchor: 'left'
      },
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: 'rgba(0,0,0,0)'
    })
  }, [pcaData, selectedPiece, pieceMetrics])

  if (!pcaData || pcaData.length === 0) {
    return (
      <Card className="w-full">
        <CardHeader>
          <CardTitle>PCA Visualization</CardTitle>
          <CardDescription>
            Interactive 3D visualization of pieces in feature space
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="text-center py-8 text-slate-500">
            No PCA data available
          </div>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>PCA Visualization</CardTitle>
        <CardDescription>
          Interactive 3D visualization of pieces in feature space
        </CardDescription>
      </CardHeader>
      <CardContent>
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.5 }}
          className="w-full h-[600px] bg-white dark:bg-slate-950 rounded-lg"
        >
          <Plot
            data={plotData}
            layout={layout}
            config={{
              responsive: true,
              displayModeBar: true,
              displaylogo: false,
            }}
            style={{ width: '100%', height: '100%' }}
          />
        </motion.div>
      </CardContent>
    </Card>
  )
}
