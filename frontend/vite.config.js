import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// `npm run dev` serves the dashboard on :5173 and forwards API calls to the backend on :8000.
// `npm run build` writes frontend/dist, which the backend serves at "/" when it exists.
const backend = "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/agent": backend,
      "/audit": backend,
      "/health": backend,
      "/sandbox": backend,
    },
  },
});
