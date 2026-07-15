#!/bin/sh
# Bootstrap-then-run entrypoint for the containerized benchmark worker.
#
# The benchmark suite (ninja-benchmark--swe-bench-controller-2 + its sibling
# mini-swe-agent checkout) is a HOST checkout bind-mounted into this container at
# the same path (TAU_BENCH_ROOT). Its host-built .venv is unusable here, so this
# script builds a container-side venv in a named volume (TAU_BENCH_VENV_DIR) with
# everything the suite needs — editable mini-swe-agent, datasets, the evaluator's
# requirements, the docker SDK — plus a static docker CLI for the suite's
# docker-out-of-docker calls. Idempotent: a populated volume boots in seconds.
#
# ISOLATION: this worker must never take the rest of the stack down. Every
# bootstrap failure (missing mount, network hiccup, bad requirements) is logged
# and retried in-process after a delay — the container stays up instead of
# crash-restart-spamming, and nothing else depends on it.
set -u

VENVDIR="${TAU_BENCH_VENV_DIR:-/opt/bench-venv}"
REPO="${TAU_BENCH_REPO_DIR:?TAU_BENCH_REPO_DIR must be set (see compose.yaml)}"
RETRY="${TAU_BENCH_BOOTSTRAP_RETRY_SECONDS:-300}"
DOCKER_CLI_VERSION="${TAU_BENCH_DOCKER_CLI_VERSION:-28.3.2}"

bootstrap() {
    if [ ! -f "$REPO/run_agent_benchmark.py" ]; then
        echo "benchmark suite not found at $REPO — is TAU_BENCH_ROOT mounted (same path as on the host)?"
        return 1
    fi
    MSWA=""
    for d in "$REPO/../mini-swe-agent" "$REPO/../ninja-benchmark-mini-swe-agent"; do
        [ -f "$d/pyproject.toml" ] && MSWA="$d" && break
    done
    if [ -z "$MSWA" ]; then
        echo "no mini-swe-agent checkout next to $REPO (expected ../mini-swe-agent or ../ninja-benchmark-mini-swe-agent)"
        return 1
    fi

    [ -x "$VENVDIR/bin/python" ] || python3 -m venv "$VENVDIR" || return 1

    # Static docker CLI (the suite shells out to `docker`; the daemon is the
    # host's, via the mounted socket). Downloaded with stdlib python: the slim
    # worker image has no curl, and keeping the image identical to every other
    # worker's is what keeps this service from adding shared build risk.
    if [ ! -x "$VENVDIR/bin/docker" ]; then
        echo "downloading static docker CLI $DOCKER_CLI_VERSION ..."
        python3 - "$VENVDIR/bin/docker" "$DOCKER_CLI_VERSION" <<'PY' || return 1
import io, os, sys, tarfile, urllib.request
dest, ver = sys.argv[1], sys.argv[2]
url = f"https://download.docker.com/linux/static/stable/x86_64/docker-{ver}.tgz"
with urllib.request.urlopen(url, timeout=600) as r:
    buf = io.BytesIO(r.read())
with tarfile.open(fileobj=buf) as tar:
    with tar.extractfile("docker/docker") as src, open(dest, "wb") as out:
        out.write(src.read())
os.chmod(dest, 0o755)
PY
    fi

    if ! "$VENVDIR/bin/python" -c "import minisweagent, datasets, docker" 2>/dev/null; then
        echo "installing benchmark suite deps into $VENVDIR ..."
        "$VENVDIR/bin/pip" install --quiet --disable-pip-version-check -e "$MSWA" datasets docker || return 1
        "$VENVDIR/bin/pip" install --quiet --disable-pip-version-check -r "$REPO/SWE-bench_Pro-os/requirements.txt" || return 1
        "$VENVDIR/bin/python" -c "import minisweagent, datasets, docker" || return 1
    fi
    return 0
}

until bootstrap; do
    echo "benchmark bootstrap failed; retrying in ${RETRY}s (the rest of the stack is unaffected)"
    sleep "$RETRY"
done

export PATH="$VENVDIR/bin:$PATH"
export TAU_BENCH_VENV_PYTHON="$VENVDIR/bin/python"
# Cache HuggingFace dataset downloads in the persistent volume too.
export HF_HOME="${HF_HOME:-$VENVDIR/hf-cache}"
echo "benchmark bootstrap OK (python: $TAU_BENCH_VENV_PYTHON, $("$VENVDIR/bin/docker" --version 2>/dev/null || echo 'docker CLI: n/a'))"

# CI/testing hook: validate the bootstrap without needing a database.
if [ "${TAU_BENCH_BOOTSTRAP_ONLY:-false}" = "true" ]; then
    echo "TAU_BENCH_BOOTSTRAP_ONLY=true — exiting after successful bootstrap"
    exit 0
fi

exec benchmark-worker
