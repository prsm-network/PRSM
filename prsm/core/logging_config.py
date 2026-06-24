"""
PRSM Logging Configuration

Centralized logging configuration for all PRSM executor modules providing:
- Consistent structured logging with structlog
- Production-ready logging configuration
- Standardized log formats and metadata
- Environment-specific logging levels
- Comprehensive audit trail for compliance
"""

import structlog
import logging
import logging.handlers
import sys
from typing import Any, Dict, Optional


# Sprint 1245 — opt-in bounded/rotated FILE logging.
_ROTATING_FILE_HANDLER_SENTINEL = "_prsm_rotating_file_handler_path"
_DEFAULT_LOG_MAX_BYTES = 100 * 1024 * 1024  # 100 MB
_DEFAULT_LOG_BACKUP_COUNT = 5


def _coerce_positive_int(raw: Any, default: int) -> int:
    """Parse a positive int from an env value; fall back to ``default`` on
    missing/garbage/non-positive (a non-positive maxBytes would DISABLE rotation
    in RotatingFileHandler — the exact unbounded-growth footgun we prevent)."""
    try:
        val = int(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return val if val > 0 else default


def configure_rotating_file_logging(environ=None) -> Optional[str]:
    """Sprint 1245 — attach an opt-in, size-bounded, rotated file handler to the
    stdlib ROOT logger when ``PRSM_LOG_FILE`` is set.

    The proven node deployment posture (``setsid -f … >log 2>&1``) redirects
    stdout to a logfile that grows UNBOUNDED → eventually fills the disk and kills
    a long-running node. The runtime logging path is stdlib (node / settlement /
    p2p / monitoring all use ``logging.getLogger``), so a ``RotatingFileHandler``
    on the root logger captures the operationally-important logs in a bounded,
    rotated file instead.

    Contract:
      - ``PRSM_LOG_FILE`` UNSET → no-op, returns None; stdout-only behavior is
        byte-identical (this feature is strictly opt-in).
      - SET → attach the handler, return the resolved path.
      - IDEMPOTENT — a sentinel attr keeps a re-call (or restart) from
        double-attaching for the same path.
      - FAIL-SOFT — any setup error (unwritable dir, bad value) logs a warning and
        returns None rather than crash; logging must never take the node down.
      - The handler does NOT touch the root level or the existing stdout handler,
        so it only ADDS a sink. ``PRSM_LOG_LEVEL`` sets the file handler's own
        level (subtractive only — it can make the file LESS verbose than stdout,
        never more).

    Env: ``PRSM_LOG_FILE`` (path; enables it), ``PRSM_LOG_MAX_BYTES`` (default
    100MB, non-positive coerced to default), ``PRSM_LOG_BACKUP_COUNT`` (default 5),
    ``PRSM_LOG_LEVEL`` (default INFO).

    Caveat: ``RotatingFileHandler`` is not multi-process-safe — point distinct
    node processes at distinct ``PRSM_LOG_FILE`` paths (the proven single-process
    ``setsid`` daemon posture this targets is fine; concurrent writers to one path
    can race the rollover).
    """
    import os
    environ = environ if environ is not None else os.environ
    path = (environ.get("PRSM_LOG_FILE", "") or "").strip()
    if not path:
        return None

    root = logging.getLogger()
    for h in root.handlers:
        if getattr(h, _ROTATING_FILE_HANDLER_SENTINEL, None) == path:
            return path  # already attached for this path — idempotent

    try:
        max_bytes = _coerce_positive_int(
            environ.get("PRSM_LOG_MAX_BYTES"), _DEFAULT_LOG_MAX_BYTES)
        backup_count = _coerce_positive_int(
            environ.get("PRSM_LOG_BACKUP_COUNT"), _DEFAULT_LOG_BACKUP_COUNT)
        level_name = (environ.get("PRSM_LOG_LEVEL", "INFO") or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)

        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)

        handler = logging.handlers.RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8")
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)s %(name)s %(message)s"))
        setattr(handler, _ROTATING_FILE_HANDLER_SENTINEL, path)
        root.addHandler(handler)
        return path
    except Exception as exc:  # noqa: BLE001 — logging setup must never crash the node
        try:
            logging.getLogger(__name__).warning(
                "PRSM_LOG_FILE rotating-handler setup failed (%s) — "
                "continuing stdout-only", exc)
        except Exception:
            pass
        return None


def configure_structlog(
    environment: str = "production",
    log_level: str = "INFO",
    enable_json_logs: bool = True,
    log_file_path: str = None
) -> None:
    """
    Configure structlog for consistent logging across all executor modules
    
    Args:
        environment: Environment name (development, testing, production)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json_logs: Whether to output logs in JSON format for production
        log_file_path: Optional file path for log output
    """
    
    # Set up standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )
    
    # Configure processors based on environment
    processors = [
        # Add correlation IDs and timestamps
        structlog.contextvars.merge_contextvars,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        # NOTE: add_logger_name is intentionally omitted. structlog 25.x removed
        # structlog.processors.add_logger_name (referencing it raised
        # AttributeError on import) and the stdlib replacement reads logger.name,
        # which the configured PrintLoggerFactory's PrintLogger does not have (it
        # crashes at log-emit). _add_prsm_context already tolerates the absent
        # "logger" key, so dropping it restores this previously-crashing config.
        
        # Add stack info for errors
        structlog.processors.StackInfoRenderer(),
        
        # Add PRSM-specific metadata
        _add_prsm_context,
    ]
    
    if environment == "development":
        # Development: Human-readable colored output
        processors.extend([
            structlog.dev.ConsoleRenderer(colors=True)
        ])
    else:
        # Production: JSON output for log aggregation
        if enable_json_logs:
            processors.extend([
                structlog.processors.JSONRenderer()
            ])
        else:
            processors.extend([
                structlog.dev.ConsoleRenderer(colors=False)
            ])
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def _add_prsm_context(logger, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add PRSM-specific context to log entries"""
    event_dict["system"] = "PRSM"
    event_dict["version"] = "1.0.0-production"
    
    # Add component information if available
    logger_name = event_dict.get("logger", "")
    if "executor" in logger_name:
        event_dict["component"] = "executor"
    elif "orchestrator" in logger_name:
        event_dict["component"] = "orchestrator"
    elif "router" in logger_name:
        event_dict["component"] = "router"
    elif "scalability" in logger_name:
        event_dict["component"] = "scalability"
    
    return event_dict


def get_executor_logger(name: str) -> Any:
    """
    Get a standardized logger for executor modules
    
    Args:
        name: Module name (typically __name__)
        
    Returns:
        Configured structlog logger
    """
    return structlog.get_logger(name)


def log_execution_start(logger, operation: str, **kwargs) -> None:
    """Standardized execution start logging"""
    logger.info(
        "Execution started",
        operation=operation,
        **kwargs
    )


def log_execution_complete(logger, operation: str, duration: float, success: bool, **kwargs) -> None:
    """Standardized execution completion logging"""
    logger.info(
        "Execution completed",
        operation=operation,
        duration_seconds=duration,
        success=success,
        **kwargs
    )


def log_execution_error(logger, operation: str, error: Exception, duration: float = None, **kwargs) -> None:
    """Standardized execution error logging"""
    logger.error(
        "Execution failed",
        operation=operation,
        error_type=type(error).__name__,
        error_message=str(error),
        duration_seconds=duration,
        **kwargs,
        exc_info=True
    )


def log_security_event(logger, event_type: str, severity: str, **kwargs) -> None:
    """Standardized security event logging for compliance"""
    logger.warning(
        "Security event",
        event_type=event_type,
        severity=severity,
        audit_trail=True,
        **kwargs
    )


def log_performance_metrics(logger, operation: str, metrics: Dict[str, Any]) -> None:
    """Standardized performance metrics logging"""
    logger.info(
        "Performance metrics",
        operation=operation,
        metrics=metrics,
        category="performance"
    )


# Initialize logging on import for executor modules
def initialize_executor_logging(environment: str = None):
    """Initialize logging for executor modules"""
    import os
    
    # Detect environment if not specified
    if environment is None:
        environment = os.getenv("PRSM_ENVIRONMENT", "production")
    
    log_level = os.getenv("PRSM_LOG_LEVEL", "INFO")
    enable_json = os.getenv("PRSM_JSON_LOGS", "true").lower() == "true"
    
    configure_structlog(
        environment=environment,
        log_level=log_level,
        enable_json_logs=enable_json
    )


# Auto-initialize in production environments
import os
if os.getenv("PRSM_AUTO_INIT_LOGGING", "true").lower() == "true":
    initialize_executor_logging()