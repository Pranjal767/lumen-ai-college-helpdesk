import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig(({ command }) => ({
  base: command === "build" ? "/static/lumen-dist/" : "/",
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": "/src" } },
  server: {
    host: "127.0.0.1",
    proxy: { "/api": { target: "http://127.0.0.1:5000", changeOrigin: true } },
  },
  build: {
    outDir: "../backend/static/lumen-dist",
    emptyOutDir: true,
  },
}));
