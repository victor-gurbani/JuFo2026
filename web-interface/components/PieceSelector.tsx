'use client'

import React, { useState, useEffect } from 'react'
import { Search } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Input } from '@/components/ui/input'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { useAppStore } from '@/lib/store'

export function PieceSelector() {
  const { 
    corpusEntries, 
    searchQuery, 
    setSearchQuery, 
    selectedPiece, 
    setSelectedPiece,
    isLoading 
  } = useAppStore()
  
  const [filteredEntries, setFilteredEntries] = useState(corpusEntries)

  useEffect(() => {
    if (!searchQuery.trim()) {
      setFilteredEntries(corpusEntries)
      return
    }

    const query = searchQuery.toLowerCase()
    const filtered = corpusEntries.filter(entry => 
      entry.title.toLowerCase().includes(query) ||
      entry.composer.toLowerCase().includes(query)
    )
    setFilteredEntries(filtered)
  }, [searchQuery, corpusEntries])

  const handleSelectPiece = (entry: typeof corpusEntries[0]) => {
    setSelectedPiece(entry)
  }

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Select a Musical Piece</CardTitle>
        <CardDescription>
          Search and select a piece to analyze and visualize on the PCA map
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3 top-3 h-4 w-4 text-slate-400" />
          <Input
            placeholder="Search by title or composer..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
            disabled={isLoading}
          />
        </div>

        <div className="max-h-96 overflow-y-auto space-y-2">
          <AnimatePresence mode="popLayout">
            {filteredEntries.length === 0 ? (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                className="text-center py-8 text-slate-500"
              >
                No pieces found matching your search
              </motion.div>
            ) : (
              filteredEntries.slice(0, 50).map((entry, index) => (
                <motion.div
                  key={`${entry.composer}-${entry.title}-${index}`}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  transition={{ duration: 0.2, delay: index * 0.02 }}
                >
                  <Button
                    variant={selectedPiece === entry ? 'default' : 'outline'}
                    className="w-full justify-start text-left h-auto py-3 px-4"
                    onClick={() => handleSelectPiece(entry)}
                    disabled={isLoading}
                  >
                    <div className="flex flex-col items-start w-full">
                      <span className="font-semibold">{entry.title}</span>
                      <span className="text-sm opacity-70">
                        {entry.composer}
                        {entry.rating && ` • Rating: ${entry.rating.toFixed(2)}`}
                      </span>
                    </div>
                  </Button>
                </motion.div>
              ))
            )}
          </AnimatePresence>
        </div>

        {filteredEntries.length > 50 && (
          <p className="text-sm text-slate-500 text-center">
            Showing first 50 results. Use search to narrow down.
          </p>
        )}
      </CardContent>
    </Card>
  )
}
