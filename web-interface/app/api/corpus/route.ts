import { NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'
import { parse } from 'csv-parse/sync'

export async function GET() {
  try {
    // Path to the curated corpus CSV
    const csvPath = join(process.cwd(), '..', 'data', 'curated', 'solo_piano_corpus.csv')
    
    const fileContent = await readFile(csvPath, 'utf-8')
    const records = parse(fileContent, {
      columns: true,
      skip_empty_lines: true,
    })
    
    // Transform the data to match our interface
    const corpusEntries = records.map((record: any) => ({
      composer: record.composer_label || record.composer_name || 'Unknown',
      title: record.title || record.song_name || 'Untitled',
      mxl_path: record.mxl_path || record.mxl_abs_path || '',
      rating: record.rating ? parseFloat(record.rating) : null,
      instrument: record.instrument_names || record.instrument || 'Piano',
    }))
    
    return NextResponse.json({ corpusEntries })
  } catch (error) {
    console.error('Error loading corpus:', error)
    return NextResponse.json(
      { error: 'Failed to load corpus data' },
      { status: 500 }
    )
  }
}
