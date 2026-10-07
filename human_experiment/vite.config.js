import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

const projectDir = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  root: projectDir,
  base: "./",
  publicDir: resolve(projectDir, "public"),
  server: {
    fs: { strict: true, allow: [projectDir] }, // never serve files from outside human_experiment/
  },
  build: {
    outDir: resolve(projectDir, "dist"),
    emptyOutDir: true,
    assetsDir: "app", // keep Vite bundles separate from public/assets (faces, PCA data)
    rollupOptions: {
      input: {
        index: resolve(projectDir, "index.html"),
        review: resolve(projectDir, "review.html"),
      },
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.js"],
  },
});
