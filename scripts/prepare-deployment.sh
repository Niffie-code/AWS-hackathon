#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
npm ci
VITE_USE_MOCK=false npm run build
mkdir -p .deploy/web
cp web_handler.py vocab_prompt.py src/requirements.txt .deploy/
cp -R dist/. .deploy/web/
