"""``run_agent_in_container`` — run one agent on one task in a locked-down sandbox.

The reusable seam both task-solver phases (and any future caller) use. Per solve it:
stands up a per-solve LLM proxy + isolated network, starts a hardened container,
streams the work tree (checked-out repo + the agent bundle + prompt + harness) into a
writable ``/work`` tmpfs, execs the harness under a wall-clock and idle timeout,
parses the result, and tears everything down.

Injection is a **bind-mount** of a host work dir at ``/work`` (not the daemon archive
API, which refuses a read-only-rootfs container, and not exec-over-stdin, whose
stdin-EOF is unreliable across daemons). The image rootfs stays read-only while
``/work`` is a writable bind mount. The container runs as the orchestrator's own uid
so it owns the mounted tree and can write it with all caps dropped. The agent reaches
the model only through the proxy (per-solve token, no upstream key, no internet). All
sandboxes share one long-lived internal network the orchestrator joins once (see
``tau.sandbox.network`` for why it is shared, not per-solve, and for the two
transports).

Under docker-out-of-docker the work dir must live on a path the *host* daemon can
resolve, so ``TAU_SANDBOX_WORK_ROOT`` must point at a directory bind-mounted into the
orchestrator from the host at the SAME path (the compose service wires this). On a
host run the default temp dir works as-is.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from pathlib import Path

import docker
from docker.errors import NotFound
from docker.types import Ulimit

from tau.axiom import get_axiom
from tau.proxy import LLMProxy, SolveUsageSummary, UpstreamTarget

from .config import SandboxConfig
from .harness import (
    CONTAINER_AGENT,
    CONTAINER_HARNESS,
    CONTAINER_PROMPT,
    CONTAINER_REPO,
    CONTAINER_WORK,
    HARNESS_SCRIPT,
    RESULT_SENTINEL,
)
from .network import (
    HOST_GATEWAY_HOST,
    PROXY_ALIAS,
    ensure_shared_network,
    self_container,
)
from .types import (
    EXIT_AGENT_ERROR,
    EXIT_COMPLETED,
    EXIT_NO_ACTIVITY,
    EXIT_SANDBOX_ERROR,
    EXIT_TIME_LIMIT,
    EXIT_UPSTREAM_ERROR,
    AgentRunRequest,
    AgentRunResult,
)

log = logging.getLogger(__name__)

_TASK_SEED_BITS = 53
_TASK_SEED_MASK = (1 << _TASK_SEED_BITS) - 1
_VALIDATOR_TASK_SAMPLING_PARAMS: dict[str, float] = {"temperature": 0.0, "top_p": 1.0}
_SORTDIR_PRELOAD = "/opt/tau/libsortdir.so"


def run_agent_in_container(
    req: AgentRunRequest,
    *,
    client: docker.DockerClient,
    config: SandboxConfig,
    upstream: UpstreamTarget,
    image_tag: str,
) -> AgentRunResult:
    """Execute *req* in a fresh sandbox and return its outcome (never raises)."""
    start = time.monotonic()
    token = uuid.uuid4().hex[:12]
    net = None
    self_ctr = self_container(client)
    proxy = LLMProxy(
        upstream,
        bind_host="0.0.0.0",  # reachable from the sandbox over the shared network
        bind_port=0,  # OS-assigned free port
        enforced_model=req.model or config.model,
        enforced_sampling_params=_task_sampling_params(req.task_id),
        solve_budget=req.budget,
        upstream_read_timeout_seconds=config.proxy_request_timeout_seconds,
        smart_cache_routing=config.smart_cache_routing and upstream.endpoint_count > 1,
    )
    container = None
    workdir: Path | None = None
    # Run as the orchestrator's own uid so the container owns the bind-mounted tree
    # and can write it even with all caps dropped (no CAP_DAC_OVERRIDE).
    run_user = config.run_as_user or f"{os.getuid()}:{os.getgid()}"
    try:
        workdir = _prepare_workdir(req)
        proxy.start()
        run_kwargs = _hardening_kwargs(config)
        run_kwargs["user"] = run_user
        run_kwargs["volumes"] = {str(workdir): {"bind": CONTAINER_WORK, "mode": "rw"}}
        if self_ctr is not None:
            # Production: the shared internet-less network (orchestrator already joined
            # once at startup; this is a cheap idempotent re-check). Deliberately NOT a
            # per-solve network — churning the orchestrator across many internal nets
            # under concurrency breaks its own DNS/egress (proxy upstream calls fail).
            net = ensure_shared_network(client, self_ctr)
            run_kwargs["network"] = net.name
            proxy_base_url = f"http://{PROXY_ALIAS}:{proxy.port}/v1"
        else:
            # Dev: orchestrator on the host; sandbox reaches it via host-gateway.
            run_kwargs["extra_hosts"] = {HOST_GATEWAY_HOST: "host-gateway"}
            proxy_base_url = f"http://{HOST_GATEWAY_HOST}:{proxy.port}/v1"

        container = client.containers.run(
            image_tag,
            command=["sleep", str(config.container_ttl_seconds)],
            name=f"tau-sbx-{token}",
            detach=True,
            **run_kwargs,
        )
        log.info(
            "sandbox %s started for task %s; running agent", container.name, req.task_id
        )
        get_axiom().info(
            source="task-solver",
            event_type="sandbox_started",
            task_id=req.task_id,
            container_name=container.name,
        )

        exit_reason, success, patch, error = _exec_harness(
            container=container,
            proxy=proxy,
            config=config,
            req=req,
            proxy_base_url=proxy_base_url,
            user=run_user,
        )
        usage = proxy.usage_snapshot()
        # A tripped budget (the agent overspent) overrides a completed run — persist it.
        if usage.budget_exceeded_reason and exit_reason == EXIT_COMPLETED:
            exit_reason = usage.budget_exceeded_reason
            success = False
        # LLM infrastructure fault: the solve produced no usable result AND the proxy saw
        # infra-class upstream errors (unreachable / timeout / out of funds / rate limit /
        # provider 5xx). This is miner-unrelated, so surface a retryable upstream_error
        # rather than a result the caller would persist. A run that DID finish with a real
        # patch is kept, even if a transient upstream blip occurred along the way.
        elif usage.upstream_error_count > 0 and not (success and patch.strip()):
            log.warning(
                "task %s: %d/%d upstream request(s) failed on infrastructure "
                "(unreachable/timeout/funds/rate/5xx) with no usable result — "
                "reporting upstream_error (retryable)",
                req.task_id,
                usage.upstream_error_count,
                usage.request_count,
            )
            exit_reason = EXIT_UPSTREAM_ERROR
            success = False
        # A watchdog kill discards the harness result line, but the agent's edits are
        # still on the host in the bind-mounted work tree. Salvage the real git diff
        # (same tracked+untracked diff the harness itself would have emitted) so the
        # judge scores the work that was actually produced instead of an automatic
        # empty solution — otherwise a wall-clock kill against a near-finished
        # opponent swings a duel task by a full point on scheduling luck alone. The
        # tree may be mid-write at kill time; a torn diff scores low, which is still
        # strictly fairer than nothing. exit_reason/success are left as the kill.
        elif exit_reason in (EXIT_TIME_LIMIT, EXIT_NO_ACTIVITY) and workdir is not None:
            patch = _salvage_repo_diff(workdir / "repo")
            if patch.strip():
                log.info(
                    "task %s: salvaged %d-byte partial patch from %s work tree",
                    req.task_id,
                    len(patch),
                    exit_reason,
                )
                get_axiom().info(
                    source="task-solver",
                    event_type="timeout_patch_salvaged",
                    task_id=req.task_id,
                    exit_reason=exit_reason,
                    patch_bytes=len(patch),
                )
        return AgentRunResult(
            success=success,
            solution_diff=patch,
            exit_reason=exit_reason,
            elapsed_seconds=time.monotonic() - start,
            usage=usage,
            error=error,
        )
    except Exception as exc:  # noqa: BLE001 — a sandbox failure is a result, not a crash
        log.exception("sandbox run failed for task %s", req.task_id)
        return AgentRunResult(
            success=False,
            solution_diff="",
            exit_reason=EXIT_SANDBOX_ERROR,
            elapsed_seconds=time.monotonic() - start,
            usage=_safe_usage(proxy),
            error=str(exc),
        )
    finally:
        _cleanup(container, proxy)
        if workdir is not None:
            shutil.rmtree(workdir, ignore_errors=True)


def _task_seed(task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & _TASK_SEED_MASK


def _task_sampling_params(task_id: str) -> dict[str, float | int]:
    return {**_VALIDATOR_TASK_SAMPLING_PARAMS, "seed": _task_seed(task_id)}


def _deterministic_agent_env() -> dict[str, str]:
    env = {
        "HOME": "/tmp",
        "TMPDIR": "/tmp",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONUNBUFFERED": "1",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    if not os.environ.get("TAU_DISABLE_SORTDIR"):
        env["LD_PRELOAD"] = _SORTDIR_PRELOAD
    return env


def _hardening_kwargs(config: SandboxConfig) -> dict:
    kwargs: dict = {
        "mem_limit": config.memory,
        "memswap_limit": config.memswap_limit(),
        "nano_cpus": config.nano_cpus(),
        "pids_limit": config.pids_limit,
        "read_only": config.read_only_rootfs,
        "working_dir": CONTAINER_WORK,
        # /work is a writable bind mount (added by the caller); /tmp is a size-capped
        # tmpfs for scratch. The image rootfs stays read-only.
        "tmpfs": {"/tmp": f"rw,size={config.tmp_tmpfs_size},mode=1777"},
        "ulimits": [
            Ulimit(name="nofile", soft=config.nofile_limit, hard=config.nofile_limit)
        ],
        "environment": _deterministic_agent_env(),
    }
    if config.drop_caps:
        kwargs["cap_drop"] = ["ALL"]
    if config.no_new_privileges:
        kwargs["security_opt"] = ["no-new-privileges:true"]
    if config.run_as_user:
        kwargs["user"] = config.run_as_user
    return kwargs


def _prepare_workdir(req: AgentRunRequest) -> Path:
    """Assemble the host /work tree to bind-mount: repo, agent bundle, prompt, harness.

    Layout mirrors the in-container paths: ``repo/``, ``agent/`` (the submission
    bundle, entry ``agent/agent.py``), ``task.txt``, ``harness.py``. Created under
    ``TAU_SANDBOX_WORK_ROOT`` (must be host-visible under docker-out-of-docker) or the
    system temp dir on a host run.
    """
    work_root = Path(os.environ.get("TAU_SANDBOX_WORK_ROOT") or tempfile.gettempdir())
    work_root.mkdir(parents=True, exist_ok=True)
    workdir = Path(tempfile.mkdtemp(prefix="tau-sbx-", dir=work_root))
    shutil.copytree(
        req.repo_dir,
        workdir / "repo",
        symlinks=True,
        ignore_dangling_symlinks=True,
    )
    shutil.copytree(req.agent_dir, workdir / "agent")  # bundle root (agent.py + agent/)
    (workdir / "task.txt").write_text(req.problem_statement, encoding="utf-8")
    (workdir / "harness.py").write_text(HARNESS_SCRIPT, encoding="utf-8")
    return workdir


def _salvage_git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["git", "-c", "safe.directory=*", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def _salvage_repo_diff(repo_dir: Path) -> str:
    """Best-effort git diff (tracked + untracked) of a killed solve's work tree.

    Host-side twin of the harness's ``_repo_diff`` for runs the watchdog killed
    before the harness could emit its result line. Never raises: any git failure
    (missing tree, torn ``.git``, timeout) degrades to the empty diff the caller
    would have used anyway.
    """
    try:
        if not (repo_dir / ".git").exists():
            return ""
        diff = _salvage_git(["diff", "--binary", "--", "."], repo_dir).stdout or ""
        untracked = (
            _salvage_git(
                ["ls-files", "--others", "--exclude-standard", "-z"], repo_dir
            ).stdout
            or ""
        )
        for rel in [item for item in untracked.split("\0") if item]:
            file_diff = _salvage_git(
                ["diff", "--binary", "--no-index", "--", "/dev/null", rel], repo_dir
            )
            if file_diff.returncode in (0, 1):
                diff += file_diff.stdout or ""
        return diff
    except Exception:  # noqa: BLE001 — salvage is opportunistic, never fatal
        log.warning("failed to salvage work-tree diff from %s", repo_dir, exc_info=True)
        return ""


def _exec_harness(
    *,
    container,  # noqa: ANN001 — docker.models.containers.Container
    proxy: LLMProxy,
    config: SandboxConfig,
    req: AgentRunRequest,
    proxy_base_url: str,
    user: str,
) -> tuple[str, bool, str, str | None]:
    """Run the harness under a watchdog; return (exit_reason, success, patch, error).

    ``success`` means the harness ran to completion (the agent returned without
    raising); the caller decides qualification from the diff.
    """
    env = {
        **_deterministic_agent_env(),
        "TAU_REPO_DIR": CONTAINER_REPO,
        "TAU_PROMPT_FILE": CONTAINER_PROMPT,
        "TAU_AGENT_FILE": CONTAINER_AGENT,
        "AGENT_MODEL": req.model or config.model,
        "OPENAI_BASE_URL": proxy_base_url,
        "OPENAI_API_KEY": proxy.auth_token,
        "AGENT_API_BASE": proxy_base_url,
        "AGENT_API_KEY": proxy.auth_token,
        "TAU_AGENT_TIMEOUT_SECONDS": str(req.timeout_seconds or config.hard_timeout_seconds),
    }
    hard_timeout = req.timeout_seconds or config.hard_timeout_seconds
    watchdog = _Watchdog(
        container, proxy, hard_timeout, config.first_token_timeout_seconds
    )
    watchdog.start()
    try:
        _, output = container.exec_run(
            ["python3", CONTAINER_HARNESS],
            environment=env,
            workdir=CONTAINER_WORK,
            user=user,
        )
    except Exception as exc:  # noqa: BLE001 — container killed mid-exec lands here
        output, exec_error = b"", str(exc)
    else:
        exec_error = None
    finally:
        watchdog.stop()

    if watchdog.killed_reason:
        return watchdog.killed_reason, False, "", "sandbox killed by watchdog"

    payload = _parse_result(output)
    if payload is None:
        text = (
            output.decode("utf-8", errors="replace")
            if isinstance(output, bytes)
            else ""
        )
        if exec_error is not None:
            # The exec call itself failed (daemon/container problem) — sandbox infra.
            return EXIT_SANDBOX_ERROR, False, "", exec_error
        # Exec ran but the agent emitted no result line: it hard-exited (SystemExit /
        # os._exit / a native crash / OOM), bypassing the harness's try/except. That is
        # the agent's failure, not ours, so it is a (persisted) agent error.
        return EXIT_AGENT_ERROR, False, "", (text[-1000:] or "no harness output")
    if not payload.get("ok"):
        return EXIT_AGENT_ERROR, False, payload.get("patch", ""), payload.get("error")
    return EXIT_COMPLETED, True, payload.get("patch", ""), None


def _parse_result(output: bytes | None) -> dict | None:
    if not output:
        return None
    text = output.decode("utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith(RESULT_SENTINEL):
            try:
                return json.loads(line[len(RESULT_SENTINEL) :])
            except json.JSONDecodeError:
                return None
    return None


class _Watchdog:
    """Kills the sandbox if it exceeds the wall-clock cap or never gets a first token."""

    def __init__(
        self, container, proxy: LLMProxy, hard_timeout: int, first_token_timeout: int
    ):  # noqa: ANN001
        self._container = container
        self._proxy = proxy
        self._hard_timeout = hard_timeout
        self._first_token_timeout = first_token_timeout
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.killed_reason: str | None = None

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._done.set()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        from .types import EXIT_NO_ACTIVITY, EXIT_TIME_LIMIT

        start = time.monotonic()
        while not self._done.wait(2.0):
            elapsed = time.monotonic() - start
            if elapsed >= self._hard_timeout:
                self.killed_reason = EXIT_TIME_LIMIT
            elif (
                elapsed >= self._first_token_timeout
                and self._proxy.usage_snapshot().first_token_count == 0
            ):
                self.killed_reason = EXIT_NO_ACTIVITY
            if self.killed_reason:
                try:
                    self._container.kill()
                except Exception:  # noqa: BLE001 — already gone is fine
                    pass
                return


def _safe_usage(proxy: LLMProxy) -> SolveUsageSummary | None:
    try:
        return proxy.usage_snapshot()
    except Exception:  # noqa: BLE001
        return None


def _cleanup(container, proxy: LLMProxy) -> None:  # noqa: ANN001
    """Tear down the per-solve container and proxy.

    The shared internal network and the orchestrator's attachment to it are
    intentionally left in place (they are reused across solves — see
    ``tau.sandbox.network``); only the container and this solve's proxy are removed.
    """
    if container is not None:
        try:
            container.remove(force=True)
        except NotFound:
            pass
        except Exception:  # noqa: BLE001
            log.warning("failed to remove sandbox container", exc_info=True)
    try:
        proxy.stop()
    except Exception:  # noqa: BLE001
        log.warning("failed to stop proxy", exc_info=True)
