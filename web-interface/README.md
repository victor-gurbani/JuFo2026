# JuFo 2026 - Modern Web Interface

A modern presentation layer for the JuFo 2026 musical analysis project using a Hybrid Static/Client-Side SPA approach.

## Tech Stack

- **Next.js 14+**: Core framework with App Router
- **React 18**: For building UI components
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Shadcn/ui**: Accessible and customizable UI components
- **React-Plotly.js**: Interactive 3D visualizations
- **Framer Motion**: Smooth animations and transitions
- **Zustand**: Lightweight state management

## Features

- 🎵 **Piece Selection**: Search and select from the curated corpus
- 📊 **Interactive PCA Visualization**: Explore pieces in 3D feature space
- 📈 **Metrics Display**: View computed harmonic, melodic, and rhythmic features
- ✨ **Modern UI**: Smooth animations and responsive design
- 🌓 **Dark Mode**: Automatic dark mode support

## Getting Started

### Prerequisites

- Node.js 18+
- Python 3.10+ (for the analysis backend)
- The JuFo2026 data corpus

### Installation

```bash
npm install
```

### Development

Run the development server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

You can start editing the page by modifying `app/page.tsx`. The page auto-updates as you edit the file.

### Building for Production

Build the application:

```bash
npm run build
```

Start the production server:

```bash
npm start
```

### Static Export (for GitHub Pages)

To export as a static site:

1. Update `next.config.ts` to use `output: 'export'`
2. Run:

```bash
npm run build
```

The static files will be in the `out/` directory.

## Architecture

### Component Structure

```
components/
├── PieceSelector.tsx      # Search and select musical pieces
├── MetricsDisplay.tsx     # Display computed metrics
├── PCAVisualization.tsx   # Interactive 3D PCA plot
└── ui/                    # Reusable UI components
    ├── button.tsx
    ├── input.tsx
    └── card.tsx
```

### API Routes

```
app/api/
├── corpus/route.ts        # Load curated corpus data
├── analyze/route.ts       # Analyze a selected piece
└── pca/route.ts           # Load PCA visualization data
```

### State Management

Uses Zustand for global state management:
- Corpus entries
- Selected piece
- Computed metrics
- PCA data
- Loading states

## Integration with Python Backend

The web interface communicates with the Python analysis scripts through API routes that execute:

- `src/highlight_pca_piece.py` - For projecting pieces onto PCA space
- `src/harmonic_features.py` - For harmonic analysis
- `src/melodic_features.py` - For melodic analysis
- `src/rhythmic_features.py` - For rhythmic analysis

## Learn More

To learn more about Next.js, take a look at the following resources:

- [Next.js Documentation](https://nextjs.org/docs) - learn about Next.js features and API.
- [Learn Next.js](https://nextjs.org/learn) - an interactive Next.js tutorial.

## License

MIT - Same as the parent JuFo2026 project
