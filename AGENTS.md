# AGENTS.md

This document serves as the central reference for all AI coding agents, orchestrators, and subagents operating within this repository. It defines the project's architecture, conventions, and operational rules to ensure consistency and quality across all automated and manual changes.

## 1. Project Overview & Architecture

This repository is a dual-stack musicological analysis project, designed for academic research and **Jugend Forscht** submissions.

- **Python Backend (`/src`)**: Transforms raw MusicXML data into statistical insights (feature extraction, PCA, significance testing). Scripts are designed to be run individually based on the analysis stage needed.
- **Web Frontend (`/web-interface`)**: An interactive Next.js 15+ application using React 19, Tailwind CSS v4, Zustand, and Plotly.js to visualize data and explore the stylistic 3D feature space interactively.
- **Documentation/Papers**: The root directory contains LaTeX (`.tex`) files and Markdown files required for academic papers and Jugend Forscht documentation.

## 2. AI Operational & Workflow Rules

### General Directives
1. **Understand Before Modifying**: Always read the existing file content before making changes. Match the existing style.
2. **Minimal Edits**: Do not refactor code unrelated to the task. Fix minimally.
3. **File Creation**: Avoid creating new files unless explicitly instructed. Prefer modifying existing files.
4. **No Suppressions**: Never use `@ts-ignore`, `@ts-expect-error`, or `as any` in TypeScript. Never use `# type: ignore` in Python unless strictly necessary and explicitly documented.

### Git & Version Control
- **Commit Messages**: Write concise, descriptive commit messages focusing on the "why" rather than the "what". Format: `[type] description` (e.g., `feat: add PCA 3D visualization`, `fix: handle missing tempo markings`).
- **Atomic Commits**: Group related changes into a single commit.
- **Documentation**: Keep `README.md` and related LaTeX/Markdown files strictly updated when fundamental analysis, pipeline steps, or architectural changes are made. Any script added or modified to affect the core output *must* be immediately documented in `README.md` under the appropriate pipeline section.

## 3. Code Style & Architectural Guidelines

### Python (Backend/Analysis)
- **Formatting**: Follow PEP 8 guidelines. Use `snake_case` for variables and functions, `PascalCase` for classes. Private helpers must be prefixed with `_`.
- **Typing & Signatures**: Use comprehensive type hints. Include `from __future__ import annotations` at the top of files. Scripts generating statistical outputs should consistently use standard `argparse` flags (e.g., `--output`, `--cache`) and return zero on success (`sys.exit(main())`).
- **Pipeline Architecture Adherence**: New backend scripts must match the workflow and structure of existing analytical stages. They should compute results silently/cleanly and dump their outputs in appropriate structured files (JSON/CSV) mapped in `data/stats/` or `figures/`, not just console logs. Furthermore, any pipeline modification MUST be integrated identically into the central executable script (`quickstart.sh`).
- **Data Structures**: Heavily utilize `@dataclass` for structuring result objects, typically including a `.to_dict()` method for easy serialization.
- **Data Handling**: Use `pandas` for tabular data and `numpy` for numerical operations. 
- **Music Analysis**: The project relies heavily on `music21`. Strictly follow its established object models (`Scores`, `Parts`, `Measures`, `Notes`).
- **Visualization**: Use `matplotlib`, `seaborn`, or `plotly` as appropriate. Save generated figures to the `/figures` directory.
- **Error Handling**: Avoid bare `except:` blocks. Target specific exceptions (e.g., `ValueError`, `FileNotFoundError`). Use the **`skip_errors` boolean parameter pattern** to conditionally halt or continuously log/skip on extraction failures.

### Next.js & React (Frontend)
- **Framework**: Next.js App Router (`/web-interface/app`). Use functional components with React Hooks. Avoid class components.
- **Styling**: Use Tailwind CSS (v4). Combine classes using `tailwind-merge` and `clsx` (via a `cn` utility if available).
- **State Management**: Use `zustand` for global state. Avoid React Context unless necessary.
- **Typing**: Use strict TypeScript typing. Define interfaces/types for all component props, state, and API responses.
- **UI Components & Visualization**: Use `lucide-react` for icons and `react-plotly.js` for rendering complex data charts.
- **Error Handling**: Handle API and rendering errors gracefully. Use Error Boundaries in Next.js (`error.tsx`) to catch unexpected runtime errors.

## 4. Build, Lint, and Execution Commands

Agents must strictly use the following commands for building, testing, and running the project.

### Web Interface (Next.js)
Always run these commands from the root directory using the `--prefix` flag, or navigate to `/web-interface` using the `workdir` parameter in bash tools.
- **Install Dependencies**: `npm run web:install`
- **Run Development Server**: `npm run web:dev` *(runs `next dev`)*
- **Build for Production**: `npm run web:build` *(runs `next build`)*
- **Linting**: `npm run web:lint` *(runs `eslint` via Next.js)*

### Python Analysis Scripts
- **Install Dependencies**: Run `./quickstart.sh` for auto-setup, or `pip install -r requirements.txt`.
- **Execution**: Python scripts in `/src` are run individually (e.g., `python src/melodic_features.py`).
- **Linting/Formatting**: Ensure code complies with PEP 8. Use `flake8` or `black` if installed in the environment.

### Testing & Verification
*Note: The project currently lacks a comprehensive automated test suite. Follow these rules for manual verification and when adding new tests.*
- **Python Verification**: Before concluding a task, run `python -m py_compile src/<modified_file>.py` to verify syntax.
- **Python Tests**: If adding tests, use `pytest`. Place them in `/tests`. Run a single test via `pytest path/to/test_file.py::test_function_name`.
- **Web Verification**: Run `npm run web:lint` to ensure frontend changes do not introduce regressions before concluding a task.
- **Web Tests**: If adding tests, use `vitest` or `jest`. Run a single test via `npx vitest run path/to/test.ts`.

## 5. Subagent Delegation Rules

Orchestrator agents must delegate tasks based on the following domain expertise:
- **`visual-engineering`**: All React, Next.js, Tailwind, and UI/UX tasks in `/web-interface`.
- **`ultrabrain` / `deep`**: Complex Python data analysis, PCA algorithms, or statistical significance tests in `/src`.
- **`librarian`**: Researching `music21` documentation, pandas operations, or academic/LaTeX formatting.
- **`quick`**: Minor typo fixes, simple configuration updates, or running standard build/lint scripts.