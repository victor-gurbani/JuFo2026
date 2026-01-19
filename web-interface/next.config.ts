import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
  output: 'standalone',
  // For development, we use server-side rendering
  // For production deployment as static site, uncomment the following:
  // output: 'export',
  // images: { unoptimized: true },
};

export default nextConfig;
