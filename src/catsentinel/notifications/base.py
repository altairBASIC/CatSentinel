"""
Abstract Base Class for Notification System.

Defines the interface for async notification dispatchers
to avoid blocking the main video processing loop.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np


@dataclass
class NotificationPayload:
    """Payload for detection notifications."""
    
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    image: Optional[np.ndarray] = None
    detection_count: int = 0
    confidence: float = 0.0
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary (excluding image)."""
        return {
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "detection_count": self.detection_count,
            "confidence": self.confidence,
            "metadata": self.metadata
        }


class Notifier(ABC):
    """
    Abstract base class for notification dispatchers.
    
    All notifiers must implement async send_alert() to avoid
    blocking the main video processing loop.
    
    Usage:
        notifier = TelegramNotifier(bot_token, chat_id)
        await notifier.send_alert(payload)
    """
    
    def __init__(self, enabled: bool = True, cooldown_seconds: float = 30.0):
        """
        Initialize notifier.
        
        Args:
            enabled: Whether this notifier is active.
            cooldown_seconds: Minimum time between notifications.
        """
        self.enabled = enabled
        self.cooldown_seconds = cooldown_seconds
        self._last_notification: Optional[datetime] = None
    
    @property
    def can_send(self) -> bool:
        """Check if cooldown has elapsed since last notification."""
        if not self.enabled:
            return False
        
        if self._last_notification is None:
            return True
        
        elapsed = (datetime.now() - self._last_notification).total_seconds()
        return elapsed >= self.cooldown_seconds
    
    @abstractmethod
    async def send_alert(self, payload: NotificationPayload) -> bool:
        """
        Send an alert notification asynchronously.
        
        Args:
            payload: Notification payload with message and optional image.
        
        Returns:
            True if notification was sent successfully.
        """
        pass
    
    @abstractmethod
    async def test_connection(self) -> bool:
        """
        Test if the notification service is reachable.
        
        Returns:
            True if connection test passed.
        """
        pass
    
    def _update_last_notification(self) -> None:
        """Update the last notification timestamp."""
        self._last_notification = datetime.now()
