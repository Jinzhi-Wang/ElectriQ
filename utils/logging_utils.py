"""
Logging Utilities Module for ElectriQ
Configures structured logging, file rotation, and console output.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union, Dict, Any
from datetime import datetime
import json
import os

# Try importing rich for better console output
try:
    from rich.logging import RichHandler
    from rich.console import Console

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


class JsonFormatter(logging.Formatter):
    """
    Formatter that outputs JSON strings for structured logging.
    Useful for log aggregation systems (ELK, Splunk).
    """

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.utcfromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }

        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)

        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['args', 'asctime', 'created', 'exc_info', 'exc_text',
                           'filename', 'funcName', 'levelname', 'levelno', 'lineno',
                           'module', 'msecs', 'message', 'msg', 'name', 'pathname',
                           'process', 'processName', 'relativeCreated', 'stack_info',
                           'thread', 'threadName']:
                try:
                    json.dumps(value)  # Check if serializable
                    log_data[key] = value
                except TypeError:
                    log_data[key] = str(value)

        return json.dumps(log_data)


def setup_logging(
        log_level: Union[str, int] = logging.INFO,
        log_file: Optional[Union[str, Path]] = None,
        log_format: str = 'console',  # 'console', 'json', 'simple'
        enable_console: bool = True,
        rotation_max_bytes: int = 10 * 1024 * 1024,  # 10MB
        rotation_backup_count: int = 5,
        app_name: str = "ElectriQ"
) -> logging.Logger:
    """
    Configure the root logger with handlers and formatters.

    Args:
        log_level: Logging level (INFO, DEBUG, etc.)
        log_file: Path to log file. If None, no file logging.
        log_format: Format style ('console', 'json', 'simple')
        enable_console: Whether to log to stdout
        rotation_max_bytes: Max size before rotation
        rotation_backup_count: Number of backup files to keep
        app_name: Name of the application for logger

    Returns:
        logging.Logger: Configured logger instance
    """
    logger = logging.getLogger(app_name)
    logger.setLevel(log_level)

    # Clear existing handlers
    logger.handlers.clear()

    # Create formatters
    if log_format == 'json':
        formatter = JsonFormatter()
        console_formatter = logging.Formatter('%(levelname)s - %(message)s')
    elif log_format == 'simple':
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_formatter = formatter
    else:  # console (rich or standard)
        if RICH_AVAILABLE:
            # Rich handles formatting internally
            formatter = logging.Formatter('%(message)s')
            console_formatter = formatter
        else:
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            console_formatter = formatter

    # Console Handler
    if enable_console:
        if RICH_AVAILABLE and log_format == 'console':
            console_handler = RichHandler(
                rich_tracebacks=True,
                tracebacks_show_locals=log_level == logging.DEBUG,
                markup=True
            )
            console_handler.setFormatter(formatter)
        else:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(console_formatter)

        logger.addHandler(console_handler)

    # File Handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler

        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=rotation_max_bytes,
            backupCount=rotation_backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.info(f"Logging to file: {log_path}")

    # Capture warnings
    logging.captureWarnings(True)

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance.

    Args:
        name: Logger name (inherits from root if None)

    Returns:
        logging.Logger: Logger instance
    """
    if name:
        return logging.getLogger(f"ElectriQ.{name}")
    return logging.getLogger("ElectriQ")


class LogContext:
    """
    Context manager for adding temporary context to logs.
    """

    def __init__(self, logger: logging.Logger, **kwargs):
        self.logger = logger
        self.context = kwargs
        self.old_factory = logging.getLogRecordFactory()

    def _add_context(self, record: logging.LogRecord):
        for k, v in self.context.items():
            setattr(record, k, v)
        # Call original factory logic if needed, but usually just setting attrs is enough
        # The actual record creation happens before this hook in some versions,
        # so this is primarily for filters or specific adapter usage.
        # A more robust way is using LoggerAdapter.

    def __enter__(self):
        # Using LoggerAdapter is cleaner for context
        self.adapter = logging.LoggerAdapter(self.logger, self.context)
        return self.adapter

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


# Alternative: Simple Logger Adapter usage
def get_contextual_logger(logger: logging.Logger, **context) -> logging.LoggerAdapter:
    """
    Create a logger adapter with extra context.

    Usage:
        log = get_contextual_logger(get_logger(), user_id="123", session="abc")
        log.info("Message") # Automatically includes user_id and session
    """
    return logging.LoggerAdapter(logger, context)


if __name__ == '__main__':
    # Example usage
    log = setup_logging(
        log_level=logging.DEBUG,
        log_file="logs/electriq.log",
        log_format='console',
        app_name="ElectriQ_Test"
    )

    log.info("Application started")

    # Contextual logging
    ctx_log = get_contextual_logger(log, experiment_id="exp_123", model="gpt-4")
    ctx_log.info("Running evaluation step")

    try:
        1 / 0
    except ZeroDivisionError:
        log.exception("An error occurred during division")

    log.warning("Shutting down")