import atexit
import datetime
import logging
import os
import platform
import sys
import traceback

_LOG_DIR = os.path.join(os.path.expanduser("~"), ".flimkit", "logs")
_session_log_path: str | None = None
_logger: logging.Logger | None = None
_original_excepthook = None


def _get_system_info() -> str:
    """Collect system and environment info for error reports."""
    try:
        from flimkit._version import __version__
    except Exception:
        __version__ = "unknown"

    lines = [
        f"FLIMKit version : {__version__}",
        f"Python          : {sys.version}",
        f"Platform        : {platform.platform()}",
        f"OS              : {platform.system()} {platform.release()}",
        f"Machine         : {platform.machine()}",
        f"Processor       : {platform.processor() or 'N/A'}",
    ]

    # NumPy / Matplotlib versions if available
    for pkg_name in ("numpy", "matplotlib", "scipy"):
        try:
            mod = __import__(pkg_name)
            lines.append(f"{pkg_name:16s}: {mod.__version__}")
        except Exception:
            pass

    return "\n".join(lines)


def _crash_excepthook(exc_type, exc_value, exc_tb):
    """Global exception hook that logs uncaught exceptions before crashing."""
    if _logger is not None:
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        _logger.critical(
            "UNCAUGHT EXCEPTION (main thread)\n%s\n%s",
            tb_text,
            _get_system_info(),
        )
        # Force flush so log is written even if the process is killed
        for handler in _logger.handlers:
            handler.flush()

    # Call the original hook so the traceback still prints to stderr
    if _original_excepthook is not None:
        _original_excepthook(exc_type, exc_value, exc_tb)


def _thread_excepthook(args):
    """threading.excepthook — catches uncaught exceptions in any thread."""
    if _logger is not None:
        tb_text = "".join(
            traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback)
        )
        thread_name = args.thread.name if args.thread else "unknown"
        _logger.critical(
            "UNCAUGHT EXCEPTION (thread: %s)\n%s",
            thread_name,
            tb_text,
        )
        for handler in _logger.handlers:
            handler.flush()


def install_tk_error_handler(root):
    """
    Install a Tk callback exception handler on the given root window.
    Call this after the Tk root is created (e.g. in _init_ui or launch_gui).
    """
    def _tk_error_handler(exc_type, exc_value, exc_tb):
        if _logger is not None:
            tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
            _logger.error("UNCAUGHT EXCEPTION (Tk callback)\n%s", tb_text)
            for handler in _logger.handlers:
                handler.flush()
        # Still print to stderr so the user sees it
        sys.__stderr__.write("".join(traceback.format_exception(exc_type, exc_value, exc_tb)))

    root.report_callback_exception = _tk_error_handler


def init_crash_handler() -> str:
    """
    Set up session logging and install the global crash handler.

    Returns the path to the current session log file.
    """
    global _session_log_path, _logger, _original_excepthook

    os.makedirs(_LOG_DIR, exist_ok=True)

    # Create a timestamped log file for this session
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    _session_log_path = os.path.join(_LOG_DIR, f"session_{timestamp}.log")

    # Configure the logger
    _logger = logging.getLogger("flimkit.crash")
    _logger.setLevel(logging.DEBUG)
    _logger.handlers.clear()

    fh = logging.FileHandler(_session_log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    _logger.addHandler(fh)

    # Write session header
    _logger.info("SESSION START")
    _logger.info("System info:\n%s", _get_system_info())

    # Install the global exception hook
    _original_excepthook = sys.excepthook
    sys.excepthook = _crash_excepthook

    # Install the threading exception hook (catches errors in background threads)
    import threading
    threading.excepthook = _thread_excepthook

    # Log a clean shutdown if we exit normally
    atexit.register(lambda: _logger.info("SESSION END (clean exit)") if _logger else None)

    return _session_log_path


def log_event(message: str, level: str = "info"):
    """Log an application event (user action, state change, etc.)."""
    if _logger is None:
        return
    getattr(_logger, level, _logger.info)(message)


def log_exception(context: str = ""):
    """Log the current exception with optional context. Call from except blocks."""
    if _logger is None:
        return
    _logger.error(
        "CAUGHT EXCEPTION%s\n%s",
        f" ({context})" if context else "",
        traceback.format_exc(),
    )
    for handler in _logger.handlers:
        handler.flush()


def get_log_dir() -> str:
    """Return the log directory path."""
    return _LOG_DIR


def get_session_log_path() -> str | None:
    """Return the current session log file path."""
    return _session_log_path


def build_export_report(include_all_sessions: bool = False) -> str:
    """
    Build a full error report string with system info and log contents.

    Parameters
    ----------
    include_all_sessions : bool
        If True, include all session logs. Otherwise just the current one.
    """
    import glob

    sections = []
    sections.append("=" * 60)
    sections.append("FLIMKit Error Report")
    sections.append(f"Generated: {datetime.datetime.now().isoformat()}")
    sections.append("=" * 60)
    sections.append("")
    sections.append("SYSTEM INFO")
    sections.append("-" * 40)
    sections.append(_get_system_info())
    sections.append("")

    log_files = sorted(
        glob.glob(os.path.join(_LOG_DIR, "*.log")),
        key=os.path.getmtime,
        reverse=True,
    )

    if not include_all_sessions and _session_log_path:
        log_files = [f for f in log_files if f == _session_log_path]

    for log_file in log_files:
        sections.append("=" * 60)
        sections.append(f"LOG: {os.path.basename(log_file)}")
        sections.append("=" * 60)
        try:
            with open(log_file, "r", encoding="utf-8") as f:
                sections.append(f.read())
        except Exception as e:
            sections.append(f"(could not read: {e})")
        sections.append("")

    return "\n".join(sections)
