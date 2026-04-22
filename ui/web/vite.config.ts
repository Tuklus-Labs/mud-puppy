import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// IMPORTANT: DO NOT set rollupOptions.output.manualChunks.
// WebKitGTK has a TDZ crash on cross-chunk module cycles that V8 tolerates silently.
// Tested the hard way on Forge. Single main chunk is mandatory.
export default defineConfig({
  plugins: [tailwindcss(), react()],
  build: {
    outDir: "dist",
    rollupOptions: {
      output: {
        // manualChunks deliberately absent — WebKitGTK TDZ lesson
      },
    },
  },
});
