import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  serverExternalPackages: ["onnxruntime-web"],
  // Vercel's file tracer doesn't follow dynamic imports inside
  // onnxruntime-web (it lazy-loads ort-wasm-*.mjs / .wasm at runtime), so
  // explicitly fold the whole dist tree into the serverless bundle for
  // every API route that might end up running inference (the bot can also
  // be called transitively via auto-advance after any human action).
  outputFileTracingIncludes: {
    "/api/games": ["./node_modules/onnxruntime-web/dist/**/*", "./public/models/**/*"],
    "/api/games/[id]": ["./node_modules/onnxruntime-web/dist/**/*", "./public/models/**/*"],
    "/api/games/[id]/step": ["./node_modules/onnxruntime-web/dist/**/*", "./public/models/**/*"],
    "/api/games/[id]/eval": ["./node_modules/onnxruntime-web/dist/**/*", "./public/models/**/*"],
  },
};

export default nextConfig;
