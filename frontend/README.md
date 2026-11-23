# Lumen AI frontend

React and TypeScript frontend for the Flask application. API requests use same-origin `/api/v1` routes.

## Development

Start Flask at `http://127.0.0.1:5000`, then run:

```powershell
npm ci
npm run dev
```

Open the Vite URL shown in the terminal. The development server proxies API requests to Flask.

## Production build

```powershell
npm ci
npm run build
```

Vite writes production files to `../backend/static/lumen-dist/`. Flask serves that build at `/`.
