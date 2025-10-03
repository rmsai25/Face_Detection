import logging
import os
import sys
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from datetime import datetime
from typing import Optional, Dict, Any
import json
import inspect

class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON string"""
        log_entry = {
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        if hasattr(record, 'props') and isinstance(record.props, dict):
            log_entry.update(record.props)
        
        return json.dumps(log_entry, ensure_ascii=False)

class ColorFormatter(logging.Formatter):
    """Color formatter for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[41m'    # Red background
    }
    RESET = '\033[0m'
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with colors"""
        log_message = super().format(record)
        if record.levelname in self.COLORS and sys.stderr.isatty():
            return f"{self.COLORS[record.levelname]}{log_message}{self.RESET}"
        return log_message

def get_caller_info() -> Dict[str, Any]:
    """Get information about the caller function"""
    try:
        frame = inspect.currentframe()
        # Go back 2 frames: 1 for this function, 1 for the logging call
        caller_frame = frame.f_back.f_back if frame and frame.f_back else None
        if caller_frame:
            info = inspect.getframeinfo(caller_frame)
            return {
                'filename': os.path.basename(info.filename),
                'function': info.function,
                'line': info.lineno
            }
    except:
        pass
    return {}

class ContextLogger:
    """Logger with context support for additional properties"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.context = {}
    
    def add_context(self, **kwargs):
        """Add context variables to all subsequent log messages"""
        self.context.update(kwargs)
    
    def clear_context(self):
        """Clear all context variables"""
        self.context.clear()
    
    def _log_with_context(self, level: int, msg: str, *args, **kwargs):
        """Log message with context"""
        extra = kwargs.pop('extra', {})
        extra['props'] = {**self.context, **get_caller_info()}
        kwargs['extra'] = extra
        self.logger.log(level, msg, *args, **kwargs)
    
    def debug(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.DEBUG, msg, *args, **kwargs)
    
    def info(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.INFO, msg, *args, **kwargs)
    
    def warning(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.WARNING, msg, *args, **kwargs)
    
    def error(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)
    
    def critical(self, msg: str, *args, **kwargs):
        self._log_with_context(logging.CRITICAL, msg, *args, **kwargs)
    
    def exception(self, msg: str, *args, **kwargs):
        kwargs['exc_info'] = True
        self._log_with_context(logging.ERROR, msg, *args, **kwargs)

def setup_logger(
    name: str = "FaceDetectionSystem",
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_console: bool = True,
    enable_file: bool = True,
    json_format: bool = False,
    log_dir: str = "logs"
) -> ContextLogger:
    """
    Setup application logger with comprehensive configuration.
    
    Args:
        name: Logger name
        log_file: Specific log file path (optional)
        level: Logging level
        max_file_size: Maximum log file size in bytes
        backup_count: Number of backup files to keep
        enable_console: Whether to enable console logging
        enable_file: Whether to enable file logging
        json_format: Whether to use JSON format for file logs
        log_dir: Directory for log files
        
    Returns:
        Configured ContextLogger instance
    """
    # Convert string level to logging level
    log_level = getattr(logging, level.upper(), logging.INFO)
    
    # Get the base logger
    base_logger = logging.getLogger(name)
    
    # Clear any existing handlers to prevent duplicate logs
    if base_logger.hasHandlers():
        base_logger.handlers.clear()
    
    base_logger.setLevel(log_level)
    
    # Prevent propagation to root logger to avoid duplicate messages
    base_logger.propagate = False
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(module)s:%(funcName)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    simple_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    color_formatter = ColorFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    
    json_formatter = JSONFormatter()
    
    # Console handler (always enabled if requested)
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        # Use color formatter for console if not in JSON mode
        if not json_format and sys.stderr.isatty():
            console_handler.setFormatter(color_formatter)
        else:
            console_handler.setFormatter(simple_formatter)
        base_logger.addHandler(console_handler)
    
    # File handlers
    if enable_file:
        # Ensure log directory exists
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        
        # Determine log file path
        if not log_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")
        
        # Create rotating file handler for general logs
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(json_formatter if json_format else detailed_formatter)
        base_logger.addHandler(file_handler)
        
        # Create error log handler for errors only
        error_log_file = os.path.join(log_dir, f"{name}_errors.log")
        error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=max_file_size,
            backupCount=backup_count,
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(json_formatter if json_format else detailed_formatter)
        base_logger.addHandler(error_handler)
    
    # Create context logger wrapper
    context_logger = ContextLogger(name)
    
    # Log initialization message
    context_logger.info(
        f"Logger initialized - Level: {level}, "
        f"Console: {enable_console}, File: {enable_file}, "
        f"JSON: {json_format}"
    )
    
    return context_logger

def setup_module_logger(module_name: str, parent_logger: ContextLogger = None) -> ContextLogger:
    """
    Setup a module-specific logger that inherits from parent configuration.
    
    Args:
        module_name: Name of the module
        parent_logger: Parent logger instance
        
    Returns:
        Module-specific ContextLogger
    """
    if parent_logger:
        # Create child logger that inherits parent's handlers
        module_logger = ContextLogger(f"{parent_logger.logger.name}.{module_name}")
        # Use same handlers as parent but with module-specific context
        return module_logger
    else:
        return setup_logger(name=f"FaceDetectionSystem.{module_name}")

# Global logger instances
_system_logger = None
_module_loggers = {}

def get_logger(name: str = "FaceDetectionSystem") -> ContextLogger:
    """
    Get or create a logger instance.
    
    Args:
        name: Logger name
        
    Returns:
        ContextLogger instance
    """
    global _system_logger, _module_loggers
    
    if name == "FaceDetectionSystem":
        if _system_logger is None:
            _system_logger = setup_logger(name)
        return _system_logger
    else:
        if name not in _module_loggers:
            _module_loggers[name] = setup_module_logger(name, _system_logger)
        return _module_loggers[name]

def initialize_logging(
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> ContextLogger:
    """
    Initialize logging system with configuration.
    
    Args:
        config: Configuration dictionary
        **kwargs: Additional configuration parameters
        
    Returns:
        Main system logger
    """
    if config is None:
        config = {}
    
    # Merge config with kwargs
    config.update(kwargs)
    
    # Set default configuration
    default_config = {
        'name': 'FaceDetectionSystem',
        'level': os.getenv('LOG_LEVEL', 'INFO'),
        'log_dir': 'logs',
        'max_file_size': 10 * 1024 * 1024,  # 10MB
        'backup_count': 5,
        'enable_console': True,
        'enable_file': True,
        'json_format': False
    }
    
    # Update defaults with provided config
    for key, value in config.items():
        if key in default_config:
            default_config[key] = value
    
    global _system_logger
    _system_logger = setup_logger(**default_config)
    
    return _system_logger

# Default initialization
try:
    # Initialize with environment variables or defaults, but disable file logging
    logger = initialize_logging(enable_file=False, log_dir=None)
except Exception as e:
    # Fallback basic logging setup - console only
    logging.basicConfig(
        level=logging.INFO, 
        format='%(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Only use console handler
    )
    logger = ContextLogger("FaceDetectionSystem")
    logger.warning(f"Using basic console logging: {e}")

# Convenience functions for common logging patterns
def log_performance(func):
    """Decorator to log function performance"""
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        logger = get_logger("Performance")
        
        try:
            result = func(*args, **kwargs)
            execution_time = (datetime.now() - start_time).total_seconds()
            
            logger.debug(
                f"Function {func.__name__} executed in {execution_time:.3f}s",
                extra={'props': {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'module': func.__module__
                }}
            )
            return result
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.3f}s: {e}",
                extra={'props': {
                    'function': func.__name__,
                    'execution_time': execution_time,
                    'error': str(e),
                    'module': func.__module__
                }}
            )
            raise
    
    return wrapper

def log_detection_attempt(face_count: int, recognized_count: int, processing_time: float):
    """Log face detection and recognition results"""
    detection_logger = get_logger("Detection")
    detection_logger.info(
        f"Detection completed - Faces: {face_count}, Recognized: {recognized_count}",
        extra={'props': {
            'face_count': face_count,
            'recognized_count': recognized_count,
            'processing_time': processing_time,
            'recognition_rate': recognized_count / face_count if face_count > 0 else 0
        }}
    )

def log_system_startup():
    """Log system startup information"""
    system_logger = get_logger()
    system_logger.info("Face Detection System starting up", extra={'props': {
        'version': '1.0.0',
        'python_version': sys.version,
        'platform': sys.platform
    }})

def log_system_shutdown():
    """Log system shutdown information"""
    system_logger = get_logger()
    system_logger.info("Face Detection System shutting down")

if __name__ == "__main__":
    # Test the logging system
    test_logger = get_logger()
    
    test_logger.info("Testing info level")
    test_logger.debug("Testing debug level")
    test_logger.warning("Testing warning level")
    test_logger.error("Testing error level")
    
    # Test with context
    test_logger.add_context(user_id=123, session_id="abc123")
    test_logger.info("Message with context")
    test_logger.clear_context()
    
    # Test performance logging
    @log_performance
    def test_function():
        import time
        time.sleep(0.1)
        return "done"
    
    test_function()
    
    # Test detection logging
    log_detection_attempt(face_count=3, recognized_count=2, processing_time=0.15)



