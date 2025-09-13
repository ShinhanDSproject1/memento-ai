import logging

def setup_logger():
    """Configures and returns a logger instance."""
    # Get a new logger instance
    logger = logging.getLogger("memento_logger") # You can give it a custom name
    logger.setLevel(logging.INFO)

    # Prevent adding multiple handlers if the logger is re-initialized
    if not logger.handlers:
        # Create a console handler
        handler = logging.StreamHandler()
        
        # Create a formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        logger.addHandler(handler)
        
    return logger