"""Tools package namespace. Kept side-effect free: importing ``tools`` must not
load the tool stack (some subsystems import it while ``hermes_cli.config`` is
still initializing). Import concrete submodules directly."""


def check_file_requirements():
    """File tools only require terminal backend availability."""
    from .terminal_tool import check_terminal_requirements
    return check_terminal_requirements()


def __getattr__(name):  # PEP 562 — lazy submodule attribute access
    import importlib

    if name == "approval_prompt":
        import warnings

        warnings.warn(
            "Accessing tools.approval_prompt via the package namespace is "
            "deprecated; import tools.approval_prompt directly.",
            FutureWarning,
            stacklevel=2,
        )
    try:
        return importlib.import_module(f".{name}", __name__)
    except ImportError:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        ) from None


__all__ = ["check_file_requirements"]
