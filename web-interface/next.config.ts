import type { NextConfig } from "next";

import { dirname } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));

const nextConfig: NextConfig = {
  turbopack: {
    // Root Turbopack at the Next.js app directory.
    root: here,
  },
};

export default nextConfig;
