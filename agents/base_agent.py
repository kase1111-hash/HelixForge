"""Base Agent class for HelixForge.

Provides the abstract base class that all agents inherit from,
with common functionality for logging, configuration, and event handling.
"""

import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional

from utils.logging import get_correlated_logger, get_logger


class BaseAgent(ABC):
    """Abstract base class for all HelixForge agents.

    Provides common functionality:
    - Logging with correlation IDs
    - Configuration management
    - Event publishing
    - Error handling
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ):
        """Initialize the agent.

        Args:
            config: Agent configuration dictionary.
            correlation_id: Optional correlation ID for tracing.
        """
        self.agent_name = self.__class__.__name__
        self.config = config or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())

        # Set up logging
        self._base_logger = get_logger(f"agents.{self.agent_name}")
        self.logger = get_correlated_logger(
            f"agents.{self.agent_name}",
            self.correlation_id,
            self.agent_name
        )

        # Event handlers
        self._event_handlers: Dict[str, List[Callable]] = {}

        self.logger.info(f"Initialized {self.agent_name}")

    @property
    @abstractmethod
    def event_type(self) -> str:
        """Event type this agent publishes (e.g., 'data.ingested')."""
        pass

    @abstractmethod
    def process(self, *args, **kwargs) -> Any:
        """Main processing method. Must be implemented by subclasses."""
        pass

    def subscribe(self, event_type: str, handler: Callable) -> None:
        """Subscribe to an event type.

        Args:
            event_type: Event type to subscribe to.
            handler: Callback function to handle the event.
        """
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
        self.logger.debug(f"Subscribed to event: {event_type}")

    def publish(self, event_type: str, payload: Dict[str, Any]) -> None:
        """Publish an event.

        Args:
            event_type: Type of event being published.
            payload: Event data.
        """
        event = {
            "event_type": event_type,
            "payload": payload,
            "timestamp": datetime.utcnow().isoformat(),
            "correlation_id": self.correlation_id,
            "agent": self.agent_name
        }

        self.logger.info(
            f"Publishing event: {event_type}",
            extra={"event": event}
        )

        # Call local handlers
        if event_type in self._event_handlers:
            for handler in self._event_handlers[event_type]:
                try:
                    handler(event)
                except Exception as e:
                    self.logger.error(f"Event handler error: {e}")

    def get_config(self, key: str, default: Any = None) -> Any:
        """Get a configuration value.

        Args:
            key: Configuration key (supports dot notation).
            default: Default value if key not found.

        Returns:
            Configuration value or default.
        """
        keys = key.split(".")
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def set_correlation_id(self, correlation_id: str) -> None:
        """Set a new correlation ID.

        Args:
            correlation_id: New correlation ID.
        """
        self.correlation_id = correlation_id
        self.logger = get_correlated_logger(
            f"agents.{self.agent_name}",
            correlation_id,
            self.agent_name
        )

    def new_correlation_id(self) -> str:
        """Generate and set a new correlation ID.

        Returns:
            The new correlation ID.
        """
        new_id = str(uuid.uuid4())
        self.set_correlation_id(new_id)
        return new_id

    def log_metric(self, name: str, value: float, tags: Optional[Dict] = None) -> None:
        """Log a metric value.

        Args:
            name: Metric name.
            value: Metric value.
            tags: Optional tags.
        """
        from utils.logging import log_metric
        log_metric(self._base_logger, name, value, tags, self.correlation_id)

    def handle_error(self, error: Exception, context: Optional[Dict] = None) -> None:
        """Handle an error with logging.

        Args:
            error: The exception that occurred.
            context: Optional context information.
        """
        self.logger.error(
            f"Error in {self.agent_name}: {str(error)}",
            extra={"error_type": type(error).__name__, "context": context},
            exc_info=True
        )


class AgentRegistry:
    """Registry for managing agent instances."""

    _instance = None
    _agents: Dict[str, BaseAgent] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._agents = {}
        return cls._instance

    def register(self, name: str, agent: BaseAgent) -> None:
        """Register an agent instance.

        Args:
            name: Agent name.
            agent: Agent instance.
        """
        self._agents[name] = agent

    def get(self, name: str) -> Optional[BaseAgent]:
        """Get an agent by name.

        Args:
            name: Agent name.

        Returns:
            Agent instance or None.
        """
        return self._agents.get(name)

    def list_agents(self) -> List[str]:
        """List all registered agent names.

        Returns:
            List of agent names.
        """
        return list(self._agents.keys())

    def clear(self) -> None:
        """Clear all registered agents."""
        self._agents.clear()


# Global registry instance
agent_registry = AgentRegistry()
