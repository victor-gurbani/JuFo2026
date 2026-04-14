import type { NextConfig } from "next";

import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const repoRoot = resolve(here, "..");

const nextConfig: NextConfig = {
  turbopack: {
    root: repoRoot,
  },
  output: "standalone",
  outputFileTracingExcludes: {
    '*': [
      '../15571083/**/*',
      '../data/**/*',
      '../src/**/*',
      '../tmp/**/*',
      '../configs/**/*'
    ]
  }
};

export default nextConfig;
