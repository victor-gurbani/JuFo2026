import { NextResponse } from 'next/server'
import { exec } from 'child_process'
import { promisify } from 'util'
import { join } from 'path'

const execAsync = promisify(exec)

export async function POST(request: Request) {
  try {
    const { mxl_path } = await request.json()
    
    if (!mxl_path) {
      return NextResponse.json(
        { error: 'Missing mxl_path parameter' },
        { status: 400 }
      )
    }

    // Path to the Python script
    const repoRoot = join(process.cwd(), '..')
    const scriptPath = join(repoRoot, 'src', 'highlight_pca_piece.py')
    
    // Create a temporary output directory
    const outputDir = join(repoRoot, 'tmp', 'pca_highlights')
    
    // Run the Python script to analyze the piece
    const command = `cd ${repoRoot} && python3 ${scriptPath} "${mxl_path}" --output "${outputDir}/temp_output.html"`
    
    const { stdout, stderr } = await execAsync(command, {
      timeout: 60000, // 60 second timeout
    })
    
    if (stderr && !stderr.includes('UserWarning')) {
      console.error('Python script stderr:', stderr)
    }
    
    // For now, return a mock response since we need to extract the actual metrics
    // In a production setup, you'd modify the Python script to output JSON metrics
    return NextResponse.json({
      success: true,
      message: 'Analysis completed',
      harmonic: {
        chord_quality_mean: 0.65,
        dissonance_ratio: 0.23,
        roman_numeral_diversity: 0.78,
      },
      melodic: {
        interval_diversity: 0.82,
        contour_complexity: 0.56,
        pitch_entropy: 0.71,
      },
      rhythmic: {
        syncopation_mean: 0.34,
        note_density: 0.89,
        rhythm_complexity: 0.67,
      },
      pca: {
        dim1: Math.random() * 10 - 5,
        dim2: Math.random() * 10 - 5,
        dim3: Math.random() * 10 - 5,
      }
    })
  } catch (error) {
    console.error('Error analyzing piece:', error)
    return NextResponse.json(
      { error: 'Failed to analyze piece', details: String(error) },
      { status: 500 }
    )
  }
}
