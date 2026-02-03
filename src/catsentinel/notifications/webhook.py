"""
Webhook Notification Implementation.

Generic webhook sender for custom integrations
(Discord, Slack, Home Assistant, etc.)
"""

import logging
from typing import Dict, Optional

from .base import Notifier, NotificationPayload

logger = logging.getLogger(__name__)


class WebhookNotifier(Notifier):
    """
    Generic webhook notifier.
    
    Example:
        notifier = WebhookNotifier(
            url="https://your-server.com/webhook",
            headers={"Authorization": "Bearer token"}
        )
        
        await notifier.send_alert(payload)
    """
    
    def __init__(
        self,
        url: str,
        enabled: bool = True,
        cooldown_seconds: float = 30.0,
        headers: Optional[Dict[str, str]] = None,
        method: str = "POST"
    ):
        """
        Initialize webhook notifier.
        
        Args:
            url: Webhook endpoint URL.
            enabled: Enable/disable notifications.
            cooldown_seconds: Minimum time between alerts.
            headers: Optional HTTP headers.
            method: HTTP method (POST, PUT).
        """
        super().__init__(enabled, cooldown_seconds)
        self.url = url
        self.headers = headers or {}
        self.method = method.upper()
        self._session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def send_alert(self, payload: NotificationPayload) -> bool:
        """
        Send alert via webhook.
        
        Args:
            payload: Notification payload.
        
        Returns:
            True if sent successfully.
        """
        if not self.can_send:
            logger.debug("Webhook notification skipped (cooldown)")
            return False
        
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False
        
        try:
            session = await self._get_session()
            
            # Build JSON payload
            json_data = payload.to_dict()
            
            async with session.request(
                self.method,
                self.url,
                json=json_data,
                headers=self.headers
            ) as response:
                success = 200 <= response.status < 300
                
                if success:
                    self._update_last_notification()
                    logger.info(f"Webhook alert sent to {self.url}")
                else:
                    logger.warning(
                        f"Webhook failed: {response.status} - "
                        f"{await response.text()}"
                    )
                
                return success
                
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
    
    async def test_connection(self) -> bool:
        """Test webhook endpoint (HEAD request)."""
        try:
            session = await self._get_session()
            
            async with session.head(self.url, headers=self.headers) as response:
                success = response.status < 500
                if success:
                    logger.info(f"Webhook endpoint reachable: {self.url}")
                return success
                
        except Exception as e:
            logger.error(f"Webhook connection test failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
