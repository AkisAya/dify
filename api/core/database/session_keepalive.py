"""
Database Session Keep-Alive Utility

Prevents database session timeout during long-running operations by sending
periodic heartbeat queries to the database server, keeping the connection active.

The key insight: Pass the explicit session object so heartbeats use that
specific connection, not thread-local magic.

Usage:
    from api.core.database.session_keepalive import SessionKeepalive, session_keepalive

    # Context manager approach
    with SessionKeepalive(session=db.session, interval=60):
        for item in large_dataset:
            db.session.add(item)
        db.session.commit()  # Connection stayed alive automatically!

    # Decorator approach
    @session_keepalive(interval=60)
    def process_large_dataset():
        for item in large_dataset:
            db.session.add(item)
        db.session.commit()
"""

import functools
import logging
import threading
from collections.abc import Callable
from typing import Any, Optional, TypeVar

from flask import has_app_context
from sqlalchemy import text

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


class SessionKeepalive:
    """
    Maintains database session connection alive during long-running operations.

    This class prevents database connection timeouts by executing periodic heartbeat
    queries (SELECT 1) on an explicit session object. The heartbeat is executed
    automatically in a background daemon thread.

    Architecture:
    - Background heartbeat thread: Periodically executes heartbeats on provided session
    - Session object passed to constructor: The exact session that stays alive
    - Result: The same session object stays alive throughout the operation

    Attributes:
        session (Any): SQLAlchemy session object to send heartbeats on.
        interval (int): Time between heartbeat queries in seconds. Default is 60.
    """

    def __init__(
        self,
        session: Any,
        interval: int = 60,
    ):
        """
        Initialize SessionKeepalive.

        Args:
            session (Any): SQLAlchemy session object to use for heartbeats.
                The exact session that will be kept alive.
            interval (int): Heartbeat interval in seconds. Recommended values:
                - 60 seconds: Safe for most database setups (default)
                - 30 seconds: More aggressive, for stricter timeout policies
                - 120+ seconds: Less frequent heartbeats, lower overhead

        Raises:
            ValueError: If interval is not positive.
        """
        if interval <= 0:
            raise ValueError("Interval must be positive")

        if session is None:
            raise ValueError("session parameter is required")

        self.interval = interval
        self.session = session
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._is_running = False

    def start(self) -> None:
        """
        Start the keep-alive heartbeat background thread.

        If already running, this method is a no-op. The heartbeat thread is
        created as a daemon thread, so it won't prevent the application from
        shutting down.

        The background thread will automatically execute heartbeats at the
        specified interval using the provided session or callback.

        This method is thread-safe.
        """
        if self._is_running:
            logger.debug("SessionKeepalive already running, skipping start")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="SessionKeepalive")
        self._thread.start()
        self._is_running = True
        logger.debug("SessionKeepalive started with %ss interval", self.interval)

    def stop(self) -> None:
        """
        Stop the keep-alive heartbeat thread.

        Sends a stop signal to the heartbeat thread and waits for it to finish
        (with a 5-second timeout). This method is thread-safe and can be safely
        called even if the thread is not running.
        """
        if not self._is_running:
            logger.debug("SessionKeepalive not running, skipping stop")
            return

        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

        self._is_running = False
        logger.debug("SessionKeepalive stopped")

    def _execute_heartbeat(self) -> None:
        """
        Execute a heartbeat query to keep the session alive.

        Sends a lightweight SELECT 1 query on the provided session.
        This method is called from the background thread periodically.
        """
        try:
            # Send lightweight query to database server
            self.session.execute(text("SELECT 1"))
            logger.debug("SessionKeepalive heartbeat sent successfully")
        except Exception as e:
            logger.warning("SessionKeepalive heartbeat failed: %s", e, exc_info=False)
            # Don't re-raise the exception to keep the heartbeat loop running
            # Don't rollback either, as it might interfere with main thread's transaction

    def _heartbeat_loop(self) -> None:
        """
        Background heartbeat loop that periodically executes heartbeats.

        This runs in a daemon thread and automatically sends heartbeats
        on the provided session without requiring manual intervention.

        The heartbeat is executed using the explicitly passed session object,
        keeping that specific connection alive.
        """
        while not self._stop_event.is_set():
            self._execute_heartbeat()

            # Wait for next heartbeat or stop signal
            # Using Event.wait() instead of time.sleep() allows quick shutdown
            self._stop_event.wait(self.interval)

    def __enter__(self):
        """
        Context manager entry: start the keep-alive heartbeat.

        Returns:
            SessionKeepalive: Self, for use with 'as' clause in with statement.
        """
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit: stop the keep-alive heartbeat.

        Always stops the heartbeat thread, regardless of whether an exception
        occurred in the context block.

        Args:
            exc_type: Exception type (if any).
            exc_val: Exception value (if any).
            exc_tb: Exception traceback (if any).

        Returns:
            False: Exceptions are not suppressed by this context manager.
        """
        self.stop()
        return False


def session_keepalive(
    _func: Optional[F] = None,
    *,
    interval: int = 60,
    session_getter: Optional[Callable[[], Any]] = None,
) -> Callable[[F], F] | F:
    """
    Decorator to keep database session alive during function execution.

    This decorator automatically starts a heartbeat thread before function execution
    and stops it after the function completes. The session is obtained from the
    Flask application context (db.session) if not explicitly provided.

    Args:
        _func (Optional[F]): The function to decorate (for @session_keepalive without parentheses).
        interval (int): Heartbeat interval in seconds. Default is 60.
        session_getter (Optional[Callable[[], Any]]): A callable that returns the
            SQLAlchemy session object. If None, will use lambda: db.session from
            Flask app context. This ensures the session is obtained at the right time,
            guaranteeing it's the same session used by the decorated function.

    Returns:
        Callable: Decorated function.

    Usage:
        # Without parentheses (uses default settings)
        @session_keepalive
        def process_large_dataset():
            for item in large_dataset:
                db.session.add(item)
            db.session.commit()

        # With parentheses and parameters
        @session_keepalive(interval=60)
        def process_large_dataset():
            for item in large_dataset:
                db.session.add(item)
            db.session.commit()

        # With explicit session getter
        @session_keepalive(interval=60, session_getter=lambda: db.session)
        def process_data():
            # ... your code ...
            pass

        # With custom session
        @session_keepalive(interval=60, session_getter=lambda: my_custom_session)
        def process_with_custom_session():
            # ... your code ...
            pass

    Raises:
        RuntimeError: If session_getter is None and Flask app context is not available.

    Note:
        - The decorator uses session_getter to obtain the session at runtime,
          ensuring it's the SAME session object used by the decorated function
        - The heartbeat only sends SELECT 1 queries and does not commit or rollback
        - Transaction control remains with the decorated function
        - Using session_getter instead of direct session ensures consistency
    """

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Determine the session getter
            actual_session_getter = session_getter
            if actual_session_getter is None:
                # Check Flask app context availability
                if not has_app_context():
                    raise RuntimeError(
                        "No Flask application context available. "
                        "Either run within Flask app context or provide explicit session_getter parameter."
                    )

                # Import db here to avoid circular imports
                from extensions.ext_database import db

                # Create a getter that returns db.session
                def actual_session_getter() -> Any:
                    return db.session

            # Get the session that will be used by the function
            target_session = actual_session_getter()

            # Start keepalive with that specific session
            keepalive = SessionKeepalive(session=target_session, interval=interval)
            keepalive.start()

            try:
                # Execute the function
                result = func(*args, **kwargs)
                return result
            finally:
                # Always stop keepalive
                keepalive.stop()

        return wrapper  # type: ignore[return-value]

    # Support both @session_keepalive and @session_keepalive(...)
    if _func is None:
        # Called with parentheses: @session_keepalive(...) 
        return decorator
    else:
        # Called without parentheses: @session_keepalive
        return decorator(_func)
