import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Dev only: proxy the API to the Python server. In production Python serves dist/ itself.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8000" } },
});
