import type { NextConfig } from "next";

import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  turbopack: {
    // Prevent Next.js from inferring an unrelated monorepo root
    // (which can break file tracing on some symlinks).
    root: here,
  },
};

export default nextConfig;
