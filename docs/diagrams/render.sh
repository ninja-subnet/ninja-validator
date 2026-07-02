#!/usr/bin/env bash
# Render every .mmd in this directory to a same-named .png using the
# mermaid-cli Docker image (no local Node/Chromium needed).
#
#   ./render.sh            # render all *.mmd
#   ./render.sh foo.mmd    # render a single file
#
# Requires: docker, and the minlag/mermaid-cli image (auto-pulled on first run).
set -euo pipefail
cd "$(dirname "$0")"

IMAGE="minlag/mermaid-cli"
CFG="puppeteer-config.json"

docker image inspect "$IMAGE" >/dev/null 2>&1 || docker pull "$IMAGE"

files=("$@")
if [ ${#files[@]} -eq 0 ]; then
    mapfile -t files < <(ls *.mmd)
fi

failed=()
for src in "${files[@]}"; do
    out="${src%.mmd}.png"
    echo "rendering $src -> $out"
    # Run as root so Chromium can write into the bind mount, then reclaim ownership.
    if ! docker run --rm -u 0:0 -v "$PWD:/data" "$IMAGE" \
        -i "/data/$src" -o "/data/$out" -p "/data/$CFG" -b white -s 2 >/dev/null 2>&1; then
        echo "  !! FAILED: $src"
        failed+=("$src")
    fi
done

# Reclaim ownership of anything root just wrote.
docker run --rm -u 0:0 -v "$PWD:/data" --entrypoint chown "$IMAGE" \
    -R "$(id -u):$(id -g)" /data >/dev/null 2>&1 || true

echo "done: $(ls -1 *.png 2>/dev/null | wc -l) png(s)"
if [ ${#failed[@]} -gt 0 ]; then
    echo "FAILED (${#failed[@]}): ${failed[*]}"
    exit 1
fi
