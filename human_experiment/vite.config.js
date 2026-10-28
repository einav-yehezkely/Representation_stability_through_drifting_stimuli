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
    // The 11,817 face images (and their copy in dist/) never change while developing; watching
    // them kept the dev server busy and made every face take seconds to load. Consequence: after
    // `npm run prepare-assets` changes the faces, restart `npm run dev` so it serves the new files.
    watch: { ignored: ["**/dist/**", "**/public/assets/faces/**", "**/test-results/**"] },
  },
  build: {
    outDir: resolve(projectDir, "dist"),
    emptyOutDir: true,
    assetsDir: "app", // keep Vite bundles separate from public/assets (faces, PCA data)
    rollupOptions: {
      input: {
        index: resolve(projectDir, "index.html"),
        review: resolve(projectDir, "review.html"),
        admin: resolve(projectDir, "admin.html"),
      },
    },
  },
  test: {
    environment: "node",
    include: ["tests/**/*.test.js"],
  },
});
