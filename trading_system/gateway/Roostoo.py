'''
Gateway class for interacting with the Roostoo API

@ Michael Teguh Laksana 15 March 2026 16:00
'''


import hmac
import hashlib
import time
import httpx
import logging
from typing import Dict, Any, Optional

# Configure logging for the interface
logger = logging.getLogger("RoostooInterface")

class RoostooClient:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://api.roostoo.com"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=10.0)

    async def close(self):
        """Close the underlying HTTP client session."""
        await self.client.aclose()

    def _generate_signature(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """
        Creates the HMAC SHA256 signature required by the API.
        The message format is usually: timestamp + method + path + body
        """
        message = f"{timestamp}{method.upper()}{path}{body}"
        signature = hmac.new(
            self.api_secret,
            message.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature

    async def _request(self, method: str, path: str, params: Optional[Dict] = None, json_data: Optional[Dict] = None) -> Dict[str, Any]:
        """Core request handler with authentication headers."""
        url = f"{self.base_url}{path}"
        timestamp = str(int(time.time() * 1000))
        
        # Prepare body string for signature if it's a POST request
        body_str = ""
        if json_data:
            # Note: Ensure JSON keys are sorted or consistent if the API requires it
            import json
            body_str = json.dumps(json_data, separators=(',', ':'))

        headers = {
            "Content-Type": "application/json",
            "X-ROOSTOO-APIKEY": self.api_key,
            "X-ROOSTOO-TIMESTAMP": timestamp,
            "X-ROOSTOO-SIGNATURE": self._generate_signature(timestamp, method, path, body_str)
        }

        try:
            response = await self.client.request(
                method, url, params=params, content=body_str, headers=headers
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"API Error ({e.response.status_code}): {e.response.text}")
            return {"error": e.response.status_code, "message": e.response.text}
        except Exception as e:
            logger.error(f"Connection Error: {str(e)}")
            return {"error": "connection_failed", "message": str(e)}

    # --- Public API Methods ---

    async def get_market_data(self) -> Dict[str, Any]:
        """Fetch real-time prices for all listed coins."""
        return await self._request("GET", "/v1/market/quotes")

    async def get_account_balance(self) -> Dict[str, Any]:
        """Fetch current portfolio holdings and buying power."""
        return await self._request("GET", "/v1/account/balance")

    async def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a trade.
        side: 'BUY' or 'SELL'
        order_type: 'MARKET' or 'LIMIT'
        """
        data = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": str(quantity)
        }
        if price:
            data["price"] = str(price)
            
        return await self._request("POST", "/v1/order/place", json_data=data)

    async def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """Check the status of a specific order."""
        return await self._request("GET", f"/v1/order/status/{order_id}")

