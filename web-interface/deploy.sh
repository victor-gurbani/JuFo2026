#!/bin/bash
set -e

# Change directory to the web-interface folder
cd "$(dirname "$0")"

echo "====================================="
echo "Building the Next.js app..."
echo "====================================="
npm run build

echo "====================================="
echo "Syncing standalone files to VPS..."
echo "====================================="
# Notice the path changed: it used to be .next/standalone/JuFo2026/web-interface/
# but now it's just .next/standalone/web-interface/
rsync -avz .next/standalone/web-interface/ oracle2:/var/www/jufo2026/web-interface/

echo "====================================="
echo "Syncing static and public files..."
echo "====================================="
rsync -avz .next/static/ oracle2:/var/www/jufo2026/web-interface/.next/static/
rsync -avz public/ oracle2:/var/www/jufo2026/web-interface/public/

echo "====================================="
echo "Syncing configs..."
echo "====================================="
rsync -avz ../configs/ oracle2:/var/www/jufo2026/configs/

echo "====================================="
echo "Restarting Node server via PM2..."
echo "====================================="
ssh oracle2 "cd /var/www/jufo2026/web-interface && pm2 restart 0"

echo "====================================="
echo "Deployment completed successfully! ✅"
echo "====================================="
