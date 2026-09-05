"""CLI entry point for the hermes-agent ACP adapter.

Loads ``~/.hermes/.env``, routes logging to stderr (stdout is reserved for ACP
JSON-RPC), and starts the ACP agent server.

Usage::

    python -m acp_adapter.entry   # or: hermes acp / hermes-acp
"""

# IMPORTANT: hermes_bootstrap must be the very first import — UTF-8 stdio
# on Windows.  No-op on POSIX.  See hermes_bootstrap.py for full rationale.
try:
    import hermes_bootstrap  # noqa: F401
except ModuleNotFoundError:
    # Partial ``hermes update`` (git-reset landed, ``uv pip install -e .`` did not):
    # UTF-8 stdio setup is skipped on Windows; POSIX is unaffected.
    pass
else:
    # Stop a ``utils/``/``proxy/``/``ui/`` package in the launch cwd from shadowing Hermes modules.
    hermes_bootstrap.harden_import_path()

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path
from hermes_constants import get_hermes_home


# Liveness-probe methods outside the ACP schema. The router correctly answers JSON-RPC -32601
# (clients treat that as "agent alive"), but the dispatching supervisor task also logs
# ``"Background task failed"`` with a traceback every probe. Keep the response; silence the noise.
_BENIGN_PROBE_METHODS = frozenset({"ping", "health", "healthcheck"})


class _BenignProbeMethodFilter(logging.Filter):
    """Suppress acp 'Background task failed' tracebacks caused by unknown liveness-probe methods
    (e.g. ``ping``); every other background-task error, incl. method_not_found for non-probe
    methods, stays visible."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.getMessage() != "Background task failed" or not record.exc_info:
            return True
        # Lazy import keeps this module importable without ``agent-client-protocol``.
        try:
            from acp.exceptions import RequestError
        except ImportError:
            return True
        exc = record.exc_info[1]
        if not isinstance(exc, RequestError) or getattr(exc, "code", None) != -32601:
            return True
        data = getattr(exc, "data", None)
        return not (isinstance(data, dict) and data.get("method") in _BENIGN_PROBE_METHODS)


def _setup_logging() -> None:
    """Route all logging to stderr so stdout stays clean for ACP stdio."""
    from agent.redact import RedactingFormatter

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(RedactingFormatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                                            datefmt="%Y-%m-%d %H:%M:%S"))
    handler.addFilter(_BenignProbeMethodFilter())
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)
    for noisy in ("httpx", "httpcore", "openai"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _load_env() -> None:
    """Load .env from HERMES_HOME (default ``~/.hermes``)."""
    from hermes_cli.env_loader import load_hermes_dotenv

    hermes_home = get_hermes_home()
    loaded = load_hermes_dotenv(hermes_home=hermes_home)
    log = logging.getLogger(__name__)
    for env_file in loaded or ():
        log.info("Loaded env from %s", env_file)
    if not loaded:
        log.info("No .env found at %s, using system env", hermes_home / ".env")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="hermes-acp", description="Run Hermes Agent as an ACP stdio server.")
    parser.add_argument("--version", action="store_true", help="Print Hermes version and exit")
    parser.add_argument("--check", action="store_true", help="Verify ACP dependencies and adapter imports, then exit")
    parser.add_argument("--setup", action="store_true",
                        help="Run interactive Hermes provider/model setup for ACP terminal auth")
    parser.add_argument("--setup-browser", action="store_true",
                        help="Install agent-browser + Playwright Chromium into ~/.hermes/node/ "
                             "for browser tool support. Idempotent.")
    parser.add_argument("--yes", "-y", action="store_true", dest="assume_yes",
                        help="Accept all prompts (currently used by --setup-browser to skip the "
                             "~400 MB Chromium download confirmation).")
    parser.add_argument("--target", "-t", dest="acp_target", default=None, metavar="TARGET",
                        help="Named execution target for terminal/file/code-exec tools "
                             "(defined under terminal.targets in config.yaml). Defaults to "
                             "terminal.default_target.")
    return parser.parse_args(argv)


def _print_version() -> None:
    from hermes_cli import __version__ as hermes_version

    print(hermes_version)


def _run_check(target: str | None = None) -> None:
    import acp  # noqa: F401
    from acp_adapter.server import HermesACPAgent  # noqa: F401

    print("Hermes ACP check OK")
    # Surface which target a launched server would dispatch under, so
    # `hermes acp --check -t <name>` verifies resolution before an editor
    # depends on it. An invalid target fails closed here too (exit 2).
    try:
        from tools.execution_targets import resolve_execution_target

        resolution = resolve_execution_target(target)
        print(f"Execution target: {resolution.target} (backend={resolution.backend})")
    except Exception as exc:
        # Non-fatal for dependency verification, EXCEPT an explicitly
        # requested target that cannot resolve: fail closed so a broken
        # editor command is caught at check time.
        if target:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)


def _run_setup() -> None:
    from hermes_cli.main import main as hermes_main

    old_argv = sys.argv[:]
    try:
        sys.argv = [old_argv[0] if old_argv else "hermes", "model"]
        hermes_main()
    finally:
        sys.argv = old_argv

    # Terminal auth is the first-run UX for registry installs, so offer the browser-tools
    # install here. Skip silently without a TTY.
    if not sys.stdin.isatty():
        return
    try:
        reply = input("\nInstall browser tools? Downloads agent-browser (npm) and "
                      "optionally Playwright Chromium (~400 MB). [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return
    if reply in {"y", "yes"}:
        _run_setup_browser(assume_yes=False)


_SETUP_BROWSER_STEPS = (
    ("node", "Node.js installation failed — cannot proceed with browser tools."),
    ("browser", "Browser tools installation failed."),
)


def _run_setup_browser(assume_yes: bool = False) -> int:
    """Bootstrap agent-browser + Chromium via dep_ensure -> install.{sh,ps1}
    --ensure (shared with the runtime lazy installer). Returns 0 on success, 1 on failure."""
    from hermes_cli.dep_ensure import ensure_dependency

    try:
        for dep, failure_msg in _SETUP_BROWSER_STEPS:
            if not ensure_dependency(dep, interactive=not assume_yes):
                print(failure_msg, file=sys.stderr)
                return 1
        return 0
    except OSError as exc:
        print(f"Browser bootstrap failed: {exc}", file=sys.stderr)
        return 1


def _apply_execution_target(target: str | None) -> None:
    """Pin the process-wide execution target for this ACP server.

    ``hermes acp -t <name>`` overrides ``terminal.default_target`` for every
    tool dispatch in this process (terminal, file, code-exec, delegation).
    Validation happens here, before the JSON-RPC loop starts, so editors
    surface a clean launch error instead of per-tool failures mid-session.

    The registration path mirrors the classic CLI (``cli.py``):
    ``set_execution_target_config_source`` is the entry-point authority hook
    that ``tools.execution_targets`` resolves against. Without it, ACP is the
    one entry point whose default resolution can't be overridden per launch.
    """
    if not target:
        return

    from hermes_cli.config import load_config
    from tools.execution_targets import (
        ExecutionTargetError,
        resolve_execution_target,
        set_execution_target_config_source,
    )

    # Fail fast with the available-targets message before anything else runs.
    try:
        resolution = resolve_execution_target(target)
    except ExecutionTargetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    effective = load_config()
    terminal = effective.setdefault("terminal", {})
    if not isinstance(terminal, dict):
        raise SystemExit(
            "Error: config.yaml 'terminal' section is not a mapping; "
            "cannot apply execution target."
        )
    terminal["default_target"] = target

    # Re-validate the merged config (in case load_config() flattened or
    # normalized terminal settings differently from the raw read) and register
    # it as the dispatch authority. resolve_execution_target on the merged
    # view must succeed for the same target before the server accepts work.
    try:
        resolve_execution_target(target, config=effective)
    except ExecutionTargetError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    set_execution_target_config_source(effective)
    logger = logging.getLogger(__name__)
    logger.info(
        "ACP execution target pinned to %r (backend=%s)",
        target, resolution.backend,
    )


def main(argv: list[str] | None = None) -> None:
    """Entry point: load env, configure logging, run the ACP agent."""
    args = _parse_args(argv)
    for flag, action in (("version", _print_version), ("check", lambda: _run_check(getattr(args, "acp_target", None))), ("setup", _run_setup)):
        if getattr(args, flag):
            return action()
    if args.setup_browser:
        if rc := _run_setup_browser(assume_yes=args.assume_yes):
            sys.exit(rc)
        return

    _setup_logging()
    _load_env()
    _apply_execution_target(getattr(args, "acp_target", None))

    logger = logging.getLogger(__name__)
    logger.info("Starting hermes-agent ACP adapter")

    # Ensure the project root is on sys.path so ``from run_agent import AIAgent`` works
    project_root = str(Path(__file__).resolve().parent.parent)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    import acp
    from .server import HermesACPAgent

    # MCP discovery from config.yaml runs in a background daemon thread so the ACP server is
    # responsive immediately (blocking here cost 2-5 s); per-session MCP servers registered via
    # asyncio.to_thread are unaffected. Metadata-only hosts can opt out of the global startup.
    # Previously this blocked asyncio.run() for 2-5 s. (ACP also registers per-session MCP servers
    # dynamically via asyncio.to_thread inside the event loop; that path is unaffected.)  Moved from
    # model_tools.py module scope to avoid freezing the gateway's loop on lazy import (#16856).
    if os.environ.get("HERMES_ACP_SKIP_CONFIGURED_MCP", "").strip() != "1":
        try:
            from hermes_cli.mcp_startup import start_background_mcp_discovery

            start_background_mcp_discovery(logger=logger, thread_name="acp-mcp-discovery")
        except Exception:
            logger.debug("MCP tool discovery failed at ACP startup", exc_info=True)

    agent = HermesACPAgent()
    try:
        asyncio.run(acp.run_agent(agent, use_unstable_protocol=True))
    except KeyboardInterrupt:
        logger.info("Shutting down (KeyboardInterrupt)")
    except Exception:
        logger.exception("ACP agent crashed")
        sys.exit(1)


if __name__ == "__main__":
    main()
