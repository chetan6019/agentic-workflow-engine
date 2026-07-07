import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev server proxies /v1 to the FastAPI backend so the SPA is same-origin
// (no CORS in dev; SSE proxies fine over http/1.1). Prod uses nginx instead.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/v1": {
        target: process.env.VITE_API_URL ?? "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
