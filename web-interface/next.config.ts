import type { NextConfig } from "next";

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..", "..");

const nextConfig: NextConfig = {
  turbopack: {
    // Root Turbopack at the Next.js app directory.
    root: repoRoot,
  },
};

export default nextConfig;
