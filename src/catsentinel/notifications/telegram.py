"""
Telegram Notification Implementation.

Sends async alerts via Telegram Bot API without blocking
the main video processing loop.
"""

import io
import logging
from typing import Optional

import cv2
import numpy as np

from .base import Notifier, NotificationPayload

logger = logging.getLogger(__name__)


class TelegramNotifier(Notifier):
    """
    Telegram bot notifier using aiohttp.
    
    Example:
        notifier = TelegramNotifier(
            bot_token="123456:ABC-DEF...",
            chat_id="-100123456789"
        )
        
        payload = NotificationPayload(
            message="Cat detected on table!",
            image=frame,
            confidence=0.95
        )
        
        await notifier.send_alert(payload)
    """
    
    API_BASE = "https://api.telegram.org/bot"
    
    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        cooldown_seconds: float = 30.0,
        send_photo: bool = True
    ):
        """
        Initialize Telegram notifier.
        
        Args:
            bot_token: Telegram Bot API token.
            chat_id: Target chat/channel ID.
            enabled: Enable/disable notifications.
            cooldown_seconds: Minimum time between alerts.
            send_photo: Whether to include detection frame.
        """
        super().__init__(enabled, cooldown_seconds)
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.send_photo = send_photo
        self._session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            import aiohttp
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def send_alert(self, payload: NotificationPayload) -> bool:
        """
        Send alert via Telegram.
        
        Args:
            payload: Notification with message and optional image.
        
        Returns:
            True if sent successfully.
        """
        if not self.can_send:
            logger.debug("Telegram notification skipped (cooldown)")
            return False
        
        try:
            import aiohttp
        except ImportError:
            logger.error("aiohttp not installed. Run: pip install aiohttp")
            return False
        
        try:
            session = await self._get_session()
            
            if self.send_photo and payload.image is not None:
                success = await self._send_photo(session, payload)
            else:
                success = await self._send_message(session, payload.message)
            
            if success:
                self._update_last_notification()
                logger.info(f"Telegram alert sent: {payload.message[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Telegram notification failed: {e}")
            return False
    
    async def _send_message(self, session, message: str) -> bool:
        """Send text message."""
        url = f"{self.API_BASE}{self.bot_token}/sendMessage"
        data = {
            "chat_id": self.chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        async with session.post(url, json=data) as response:
            return response.status == 200
    
    async def _send_photo(
        self,
        session,
        payload: NotificationPayload
    ) -> bool:
        """Send photo with caption."""
        url = f"{self.API_BASE}{self.bot_token}/sendPhoto"
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', payload.image, 
                                  [cv2.IMWRITE_JPEG_QUALITY, 85])
        photo_bytes = io.BytesIO(buffer.tobytes())
        photo_bytes.name = 'detection.jpg'
        
        import aiohttp
        data = aiohttp.FormData()
        data.add_field('chat_id', self.chat_id)
        data.add_field('caption', payload.message)
        data.add_field('parse_mode', 'HTML')
        data.add_field('photo', photo_bytes, 
                       filename='detection.jpg',
                       content_type='image/jpeg')
        
        async with session.post(url, data=data) as response:
            return response.status == 200
    
    async def test_connection(self) -> bool:
        """Test Telegram bot connection."""
        try:
            session = await self._get_session()
            url = f"{self.API_BASE}{self.bot_token}/getMe"
            
            async with session.get(url) as response:
                if response.status == 200:
                    data = await response.json()
                    bot_name = data.get('result', {}).get('username', 'unknown')
                    logger.info(f"Telegram bot connected: @{bot_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False
    
    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
