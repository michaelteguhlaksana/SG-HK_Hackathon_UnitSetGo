import hmac
import hashlib
import time
import httpx
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("RoostooV3")

class RoostooAPIError(Exception):
    """Custom exception for Roostoo API failures."""
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

class RoostooClientV3:
    def __init__(self, api_key: str, api_secret: str, base_url: str = "https://mock-api.roostoo.com"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=10.0)

        self.available_pairs = set()

    async def close(self):
        await self.client.aclose()

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Signs the parameters by:
        1. Sorting keys alphabetically.
        2. Joining them into a query string (k1=v1&k2=v2).
        3. HMAC-SHA256 signing with the secret.
        """
        # Note: We convert values to strings to match the demo's .format() logic
        query_string = '&'.join([f"{k}={params[k]}" for k in sorted(params.keys())])
        return hmac.new(
            self.api_secret,
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()

    async def _request(self, method: str, path: str, params: Optional[Dict] = None, auth: bool = False) -> Dict[str, Any]:
        """Generic request handler for v3."""
        url = f"{self.base_url}{path}"
        headers = {}
        
        # Roostoo v3 quirk: Auth parameters are passed in the payload/query string itself
        payload = params.copy() if params else {}
        
        if auth:
            # Ensure timestamp is in milliseconds as per v3 demo
            if "timestamp" not in payload:
                payload["timestamp"] = int(time.time() * 1000)
            
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = self._generate_signature(payload)

        if method.upper() == "GET":
            response = await self.client.get(url, params=payload, headers=headers)
        else:
            # v3 uses form-encoding (data) instead of JSON for POST
            response = await self.client.post(url, data=payload, headers=headers)
        
        response.raise_for_status()
        data = response.json()

        if data["Success"] or path == "/v3/serverTime" or path == "/v3/exchangeInfo":
            return response.json()
        else:
            logger.error(f"Request failed ({response.status_code}) :  {e.response.text}")
            raise RoostooAPIError(f"API Error: {data.get('ErrMsg')}", status_code=response.status_code)

    # --- Public Endpoints (No Auth Required) ---

    async def get_server_time(self):
        return await self._request("GET", "/v3/serverTime")

    async def get_exchange_info(self):
        return await self._request("GET", "/v3/exchangeInfo")

    async def get_ticker(self, pair: Optional[str] = None):
        params = {}
        if pair:
            params["pair"] = pair
        # Ticker often requires a timestamp in v3 even if not 'private'
        params["timestamp"] = int(time.time()) 
        return await self._request("GET", "/v3/ticker", params=params)

    # --- Private Endpoints (Auth Required) ---

    async def get_balance(self):
        return await self._request("GET", "/v3/balance", auth=True)

    async def place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None):
        """
        symbol: e.g., 'BTC' (class adds /USD automatically to match demo)
        side: 'BUY' or 'SELL'
        """
        payload = {
            "pair": f"{symbol.upper()}/USD",
            "side": side.upper(),
            "quantity": quantity,
        }
        if price:
            payload["type"] = "LIMIT"
            payload["price"] = price
        else:
            payload["type"] = "MARKET"
            
        return await self._request("POST", "/v3/place_order", params=payload, auth=True)

    async def cancel_order(self, symbol: str = "BTC"):
        payload = {"pair": f"{symbol.upper()}/USD"}
        return await self._request("POST", "/v3/cancel_order", params=payload, auth=True)

    async def query_order(self, order_id: Optional[int] = None):
        payload = {}
        if order_id:
            payload["order_id"] = order_id
        return await self._request("POST", "/v3/query_order", params=payload, auth=True)

    async def get_pending_count(self):
        return await self._request("GET", "/v3/pending_count", auth=True)


    # --- Response handling ---
    def handle_get_serverTime(self) -> int:
        try:
            ts = self.handle_response_error(self.get_server_time())
            return ts["ServerTime"]
            
        except Exception as e:
            logger.error("Failed to get time from server.")
    
    def _parse_coin_info (self, data : Dict[str, Any]):
        #TODO: Should this send to the DB directly or should a separate function han dle this?
        #This is likely only done on warm-up, so not blocking anything
        return
    
    def _parse_ticker_price (self, data: Dict[str, Any]):
        #TODO: Should this send to the DB directly or should a separate function han dle this?
        #Unlike vefore, this will be called often. 
        return

    def handle_get_exchange_info(self) -> tuple[bool, float]:
        try:
            data = self.get_exchange_info()
            is_running = data["IsRunning"]
            init_wallet = data["InitialWallet"]
            for pair, info in data["TradePairs"].items():
                self._parse_coin_info(info)
                self.available_pairs.add(pair)

            return (is_running, init_wallet)
        except Exception as e:
            logger.error(f"Failed to get or parse exchange info: {e}")

    def handle_get_ticker(self, pair: Optional[str] = None):
        try:
            data = self.get_ticker(pair)["Data"] #No need for the rest, checked before in _request
            for ticker, price_data in data.items():
                self._parse_ticker_price (price_data)

        except Exception as e:
            logger.error(f"Failed to get or parse ticker data: {e}")