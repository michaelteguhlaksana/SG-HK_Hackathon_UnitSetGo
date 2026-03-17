import hmac
import hashlib
import time
import httpx
import logging
from typing import Dict, Any, Optional
import db.db_manager

logger = logging.getLogger("RoostooV3")

class RoostooAPIError(Exception):
    """Custom exception for Roostoo API failures."""
    def __init__(self, message, status_code=None):
        self.message = message
        self.status_code = status_code
        
        super().__init__(self.message)

class RoostooClientV3:
    def __init__(self, api_key: str, api_secret: str, db_manager:db.db_manager.DatabaseManager = None, base_url: str = "https://mock-api.roostoo.com"):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=10.0)
        if db_manager == None:
            self.db_manager = db.db_manager.DatabaseManager()
        else:
            self.db_manager = db_manager

        self.available_pairs = set()
        self.balance = dict()
        self.market_rules = {}

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
            logger.error(f"Request failed ({response.status_code}) :  {response}")
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
        params["timestamp"] = int(time.time()* 1000) 
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

    async def cancel_order(self, order_id:Optional[int] = None, pair: Optional[str] = None):
        if order_id is not None and pair is not None:
            raise ValueError("API allows only one of 'order_id' or 'pair' to be sent, not both.")
        payload = {}
        if order_id:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            # Ensure format is 'BASE/QUOTE' if user just passes 'BTC'
            formatted_pair = pair if "/" in pair else f"{pair.upper()}/USD"
            payload["pair"] = formatted_pair

        return await self._request("POST", "/v3/cancel_order", params=payload, auth=True)

    async def query_order(
        self, 
        order_id: Optional[str | int] = None, 
        pair: Optional[str] = None, 
        pending_only: Optional[bool] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ) -> list[dict]:
        """
        Queries order history or current pending status.
        
        Constraints:
        - If order_id is sent, no other optional parameters are allowed.
        - Returns an empty list [] if no orders match (instead of raising an error).
        """
        # 1. Enforce API constraint: order_id is mutually exclusive with others
        if order_id is not None:
            if any(v is not None for v in [pair, pending_only, offset, limit]):
                raise ValueError("Roostoo API: If 'order_id' is provided, no other optional parameters are allowed.")

        payload = {
            "timestamp": str(int(time.time() * 1000))  # 13-digit mandatory
        }

        # 2. Build payload based on provided params
        if order_id: 
            payload["order_id"] = str(order_id)
        else:
            if pair: 
                payload["pair"] = f"{pair.upper()}"
            if pending_only is not None: 
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if offset is not None: 
                payload["offset"] = str(offset)
            if limit is not None: 
                payload["limit"] = str(limit)

        try:
            # 3. Execute request
            res = await self._request("POST", "/v3/query_order", params=payload, auth=True)
            return res.get("OrderMatched", [])
            
        except RoostooAPIError as e:
            # 4. Handle the "No Match" case gracefully
            # The API returns Success: False for "no order matched", but we want an empty list.
            if "no order matched" in str(e).lower():
                return []
            # Re-raise if it's a real error (Unauthorized, Server Down, etc.)
            raise e

    async def get_pending_count(self):
        return await self._request("GET", "/v3/pending_count", auth=True)


    # --- Response handling ---
    async def handle_get_serverTime(self) -> int:
        try:
            ts = await self.get_server_time()
            return ts["ServerTime"]
            
        except Exception as e:
            logger.error("Failed to get time from server.")
    
    def _parse_coin_info(self, pair: str, info: Dict[str, Any]):
        """
        Stores constraints for each pair to prevent API rejection.
        """
        self.market_rules[pair] = {
            "min_qty": float(info.get("MinQuantity", 0)),
            "price_precision": int(info.get("PricePrecision", 2)),
            "qty_precision": int(info.get("QuantityPrecision", 8)),
            "min_notional": float(info.get("MinNotional", 10.0))
        }
        logger.info(f"Rules updated for {pair}: MinQty {self.market_rules[pair]['min_qty']}")

    async def handle_get_exchange_info(self) -> tuple[bool, float]:
        try:
            data = await self.get_exchange_info()
            is_running = data["IsRunning"]
            init_wallet = data["InitialWallet"]
            for pair, info in data["TradePairs"].items():
                try:
                    self._parse_coin_info(pair, info)
                    self.available_pairs.add(pair)
                except Exception as e:
                    logger.error(f"Failed to parse data for pair {pair} : {e}")
                    continue

            return (is_running, init_wallet)
        except Exception as e:
            logger.error(f"Failed to get or parse exchange info: {e}")

    async def handle_get_ticker(self, pair: Optional[str] = None):
        try:
            data = await self.get_ticker(pair) 
            
            # Accumulate all tickers into a list
            ticker_batch = []
            for ticker_name, price_data in data.get("Data", {}).items():
                try:
                    # Parse but don't save yet
                    parsed_data = {
                        "pair": ticker_name,
                        "price": float(price_data.get("LastPrice", 0)),
                        "volume": float(price_data.get("UnitTradeValue", 0))
                    }
                    ticker_batch.append(parsed_data)
                except Exception as e:
                    logger.error(f"Failed to parse ticker {ticker_name}: {e}")
                    
            # Send them all to the database in ONE lightning-fast transaction
            if ticker_batch:
                await self.db_manager.update_tickers_batch(ticker_batch)

        except Exception as e:
            logger.error(f"Failed to fetch ticker data: {e}")

    async def handle_get_balance(self):
        try:
            response = await self.get_balance()
            self.balance = response["Wallet"]  #No need for the rest, checked before in _request
        except Exception as e:
            logger.error(f"Failed to get balance information: {e}")

    async def update_order_data(self, order_detail: Dict):
        """Passes the order dictionary directly to the SQLite manager."""
        try:
            await self.db_manager.save_order(order_detail)
            logger.info(f"Order {order_detail.get('OrderID')} updated in DB (Status: {order_detail.get('Status')})")
        except Exception as e:
            logger.error(f"Failed to save order to DB: {e}")

    async def handle_place_order(self, symbol: str, side: str, quantity: float, price: Optional[float] = None):
        response = await self.place_order(symbol, side, quantity, price) 
        for details in response["OrderDetail"]:
            try:
                await self.update_order_data(details)
            except Exception as e:
                logger.error(f"Failed to handle response for placed order {details} : {e}")

    async def handle_cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None):
        try:
            # We must await the API call itself
            response = await self.cancel_order(order_id, pair)
            canceled_ids = response.get("CanceledList", [])
            
            for oid in canceled_ids:
                # Direct DB update using the ID provided by the API
                await self.db_manager.cancel_order_by_id(oid)
                
        except Exception as e:
            logger.error(f"Failed to process cancellation: {e}")

    async def handle_query_order(self, order_id=None, pair=None, pending_only=None, offset=None, limit=None):
        res = await self.query_order(
            order_id=order_id, 
            pair=pair, 
            pending_only=pending_only, 
            offset=offset, 
            limit=limit
        )
        for order_detail in res:
            try:
                await self.update_order_data(order_detail)
            except Exception as e:
                logger.error(f"Failed to update order {order_detail}: {e}")
    