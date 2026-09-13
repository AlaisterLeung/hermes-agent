from __future__ import annotations

import multiprocessing
import os

import pytest

from tools.execution_target_registry import (
    RuntimeRegistryError,
    invalidate_runtime_registry_cache,
    registry_write_lock,
)


def _hold_registry_lock(home: str, ready, release) -> None:
    os.environ["HERMES_HOME"] = home
    invalidate_runtime_registry_cache()
    with registry_write_lock():
        ready.set()
        release.wait(10)


@pytest.mark.skipif(
    os.name == "nt",
    reason="fork-based bounded flock acceptance test",
)
def test_registry_writer_lock_is_cross_process_and_bounded(monkeypatch, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    context = multiprocessing.get_context("fork")
    ready = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_registry_lock,
        args=(str(home), ready, release),
    )
    process.start()
    try:
        assert ready.wait(5)
        with pytest.raises(RuntimeRegistryError, match="Timed out"):
            with registry_write_lock(timeout_seconds=0.15):
                pytest.fail("second writer acquired a held registry lock")
    finally:
        release.set()
        process.join(5)
        if process.is_alive():
            process.terminate()
            process.join(5)
    assert process.exitcode == 0
