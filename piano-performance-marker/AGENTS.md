# Repository Guidelines

## Project Structure & Module Organization
- `src/` — React app code: components (`src/components/*`), hooks (`src/hooks/*`), utilities (`src/utils/*`), styles (`src/styles/*`), entrypoints (`src/index.js`, `src/App.jsx`).
- `public/` — static assets and HTML; sample audio under `public/audio/`.
- `evaluations/` — JSON outputs saved via the proxy API per user (`evaluations/<username>/*.json`). Treat as runtime data.
- Tests live alongside code as `*.test.jsx`.

## Build, Test, and Development Commands
- `npm install` — install dependencies.
- `npm start` — start CRA dev server on `http://localhost:5556` (see `PORT=5556`). Includes Express proxy routes from `src/setupProxy.js`:
  - `POST /api/save-evaluation`
  - `GET /api/evaluations/:username`
  - `GET /api/last-evaluation/:username`
- `npm test` — run Jest in watch mode with React Testing Library.
- `npm run build` — production build to `build/`.

## Coding Style & Naming Conventions
- Language: React + JSX, functional components and hooks.
- Indentation: 2 spaces; line width ~100–120 chars.
- Files: components in `PascalCase.jsx` (e.g., `WaveformEditor.jsx`); helpers/utilities in `camelCase.js`.
- Variables/functions: `camelCase`; constants: `UPPER_SNAKE_CASE`.
- Styling: Tailwind CSS (`src/index.css`, `tailwind.config.js`). Prefer utility classes over ad‑hoc CSS; component‑specific CSS under `src/styles/` when needed.
- Linting: CRA ESLint presets (`react-app`). Fix warnings before PR.

## Testing Guidelines
- Frameworks: Jest + React Testing Library (`@testing-library/*`).
- Naming: `ComponentName.test.jsx` near the component.
- Scope: test user interactions and DOM output; avoid implementation details.
- Run: `npm test` (press `a` for all, `u` to update snapshots if used).

## Commit & Pull Request Guidelines
- Commits: imperative, concise subject (e.g., "Add score slider validation"); include scope when clear.
- PRs must include: summary, linked issues, test plan (commands, steps), screenshots/GIFs for UI, and notes on breaking changes.

## Security & Configuration Tips
- Proxy endpoints write to `evaluations/` without auth; do not expose this dev server to the internet.
- Add `evaluations/` to `.gitignore` for real data. Review file permissions if deploying.
- Recommended Node: Active LTS (e.g., 18+).
