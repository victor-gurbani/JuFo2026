import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'
import { parse } from 'csv-parse/sync'

export async function GET() {
  try {
    // Load the three feature CSVs and merge them
    const dataDir = join(process.cwd(), '..', 'data', 'features')
    
    const harmonicPath = join(dataDir, 'harmonic_features.csv')
    const melodicPath = join(dataDir, 'melodic_features.csv')
    const rhythmicPath = join(dataDir, 'rhythmic_features.csv')
    
    // For now, we'll create mock PCA data
    // In production, you'd run the actual PCA computation or load pre-computed results
    const mockPcaData = [
      { dim1: -2.5, dim2: 1.2, dim3: 0.8, composer_label: 'Bach', title: 'Prelude in C Major' },
      { dim1: -2.3, dim2: 1.5, dim3: 0.6, composer_label: 'Bach', title: 'Fugue in D Minor' },
      { dim1: -1.8, dim2: 0.9, dim3: 1.1, composer_label: 'Bach', title: 'Invention No. 1' },
      { dim1: 0.2, dim2: -0.5, dim3: 1.3, composer_label: 'Mozart', title: 'Piano Sonata K. 545' },
      { dim1: 0.5, dim2: -0.8, dim3: 1.0, composer_label: 'Mozart', title: 'Piano Sonata K. 331' },
      { dim1: 0.3, dim2: -0.3, dim3: 1.5, composer_label: 'Mozart', title: 'Fantasia in D Minor' },
      { dim1: 1.8, dim2: -1.2, dim3: -0.5, composer_label: 'Chopin', title: 'Nocturne Op. 9 No. 2' },
      { dim1: 2.1, dim2: -1.5, dim3: -0.3, composer_label: 'Chopin', title: 'Waltz Op. 64 No. 2' },
      { dim1: 1.9, dim2: -1.0, dim3: -0.7, composer_label: 'Chopin', title: 'Prelude Op. 28 No. 4' },
      { dim1: 2.8, dim2: 0.8, dim3: -1.2, composer_label: 'Debussy', title: 'Clair de Lune' },
      { dim1: 3.2, dim2: 1.2, dim3: -1.0, composer_label: 'Debussy', title: 'Arabesque No. 1' },
      { dim1: 2.9, dim2: 0.5, dim3: -1.5, composer_label: 'Debussy', title: 'Reverie' },
    ]
    
    return NextResponse.json({ pcaData: mockPcaData })
  } catch (error) {
    console.error('Error loading PCA data:', error)
    return NextResponse.json(
      { error: 'Failed to load PCA data' },
      { status: 500 }
    )
  }
}
