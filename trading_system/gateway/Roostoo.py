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
    def __init__(
        self, 
        api_key: str, 
        api_secret: str, 
        db_manager: db.db_manager.DatabaseManager = None, 
        base_url: str = "https://mock-api.roostoo.com"
    ):
        self.api_key = api_key
        self.api_secret = api_secret.encode('utf-8')
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=10.0)
        self.db_manager = db_manager if db_manager is not None else db.db_manager.DatabaseManager()

        self.available_pairs = set()
        # Stores { 'BTC': 0.454878, 'USD': 98389.15, ... } — Free balance only
        self.balance: Dict[str, float] = {}
        self.market_rules: Dict[str, Dict] = {}

    async def close(self):
        await self.client.aclose()

    def _generate_signature(self, params: Dict[str, Any]) -> str:
        """
        Signs the parameters by:
        1. Sorting keys alphabetically.
        2. Joining into a query string (k1=v1&k2=v2).
        3. HMAC-SHA256 with the secret.
        Returns both the signature AND the sorted query string so the POST
        body can reuse the exact same byte sequence that was signed.
        """
        query_string = '&'.join(f"{k}={params[k]}" for k in sorted(params.keys()))
        signature = hmac.new(
            self.api_secret,
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        return signature, query_string

    async def _request(
        self, 
        method: str, 
        path: str, 
        params: Optional[Dict] = None, 
        auth: bool = False
    ) -> Dict[str, Any]:
        """Generic request handler for v3."""
        url = f"{self.base_url}{path}"
        headers = {}
        payload = params.copy() if params else {}
        
        if auth:
            if "timestamp" not in payload:
                payload["timestamp"] = int(time.time() * 1000)
            
            signature, sorted_query_string = self._generate_signature(payload)
            headers["RST-API-KEY"] = self.api_key
            headers["MSG-SIGNATURE"] = signature

        if method.upper() == "GET":
            response = await self.client.get(url, params=payload, headers=headers)
        else:
            # FIX: POST body must be the exact pre-sorted query string that was signed,
            # not a re-serialised dict (which httpx might order differently).
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            if auth:
                response = await self.client.post(url, content=sorted_query_string.encode(), headers=headers)
            else:
                response = await self.client.post(url, data=payload, headers=headers)
        
        response.raise_for_status()
        data = response.json()

        if path in ("/v3/serverTime", "/v3/exchangeInfo") or data.get("Success"):
            return data
        else:
            logger.error(f"API returned failure ({path}): {data.get('ErrMsg')}")
            raise RoostooAPIError(f"API Error: {data.get('ErrMsg')}", status_code=response.status_code)

    # --- Public Endpoints ---

    async def get_server_time(self):
        return await self._request("GET", "/v3/serverTime")

    async def get_exchange_info(self):
        return await self._request("GET", "/v3/exchangeInfo")

    async def get_ticker(self, pair: Optional[str] = None):
        params = {"timestamp": int(time.time() * 1000)}
        if pair:
            params["pair"] = pair
        return await self._request("GET", "/v3/ticker", params=params)

    # --- Private Endpoints ---

    async def get_balance(self):
        return await self._request("GET", "/v3/balance", auth=True)

    async def place_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: Optional[float] = None
    ):
        payload = {
            "pair": f"{symbol.upper()}/USD",
            "side": side.upper(),
            "quantity": str(quantity),
        }
        if price is not None:
            payload["type"] = "LIMIT"
            payload["price"] = str(price)
        else:
            payload["type"] = "MARKET"
            
        return await self._request("POST", "/v3/place_order", params=payload, auth=True)

    async def cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None):
        if order_id is not None and pair is not None:
            raise ValueError("API allows only one of 'order_id' or 'pair', not both.")
        payload = {}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        elif pair is not None:
            payload["pair"] = pair if "/" in pair else f"{pair.upper()}/USD"
        return await self._request("POST", "/v3/cancel_order", params=payload, auth=True)

    async def query_order(
        self, 
        order_id: Optional[str | int] = None, 
        pair: Optional[str] = None, 
        pending_only: Optional[bool] = None,
        offset: Optional[int] = None,
        limit: Optional[int] = None
    ) -> list[dict]:
        if order_id is not None:
            if any(v is not None for v in [pair, pending_only, offset, limit]):
                raise ValueError("If 'order_id' is provided, no other optional parameters are allowed.")

        payload = {"timestamp": str(int(time.time() * 1000))}
        if order_id is not None:
            payload["order_id"] = str(order_id)
        else:
            if pair:
                payload["pair"] = pair.upper()
            if pending_only is not None:
                payload["pending_only"] = "TRUE" if pending_only else "FALSE"
            if offset is not None:
                payload["offset"] = str(offset)
            if limit is not None:
                payload["limit"] = str(limit)

        try:
            res = await self._request("POST", "/v3/query_order", params=payload, auth=True)
            return res.get("OrderMatched", [])
        except RoostooAPIError as e:
            if "no order matched" in str(e).lower():
                return []
            raise e

    async def get_pending_count(self):
        return await self._request("GET", "/v3/pending_count", auth=True)

    # --- Response Handlers ---

    async def handle_get_serverTime(self) -> int:
        try:
            ts = await self.get_server_time()
            return ts["ServerTime"]
        except Exception as e:
            logger.error(f"Failed to get server time: {e}")

    def _parse_coin_info(self, pair: str, info: Dict[str, Any]):
        """
        Stores exchange constraints for each pair.
        Field names corrected to match actual API spec:
          AmountPrecision (not QuantityPrecision)
          MiniOrder       (not MinNotional / MinQuantity)
        """
        self.market_rules[pair] = {
            # MiniOrder is a NOTIONAL minimum: qty * price > MiniOrder
            "min_notional": float(info.get("MiniOrder", 1.0)),
            "price_precision": int(info.get("PricePrecision", 2)),
            "qty_precision": int(info.get("AmountPrecision", 6)),
        }
        logger.info(f"Rules updated for {pair}: {self.market_rules[pair]}")

    async def handle_get_exchange_info(self) -> tuple[bool, Any]:
        try:
            data = await self.get_exchange_info()
            is_running = data["IsRunning"]
            init_wallet = data["InitialWallet"]
            for pair, info in data["TradePairs"].items():
                try:
                    self._parse_coin_info(pair, info)
                    self.available_pairs.add(pair)
                except Exception as e:
                    logger.error(f"Failed to parse data for pair {pair}: {e}")
            return (is_running, init_wallet)
        except Exception as e:
            logger.error(f"Failed to get or parse exchange info: {e}")

    async def handle_get_ticker(self, pair: Optional[str] = None):
        try:
            data = await self.get_ticker(pair)
            ticker_batch = []
            for ticker_name, price_data in data.get("Data", {}).items():
                try:
                    ticker_batch.append({
                        "pair": ticker_name,
                        "price": float(price_data.get("LastPrice", 0)),
                        "volume": float(price_data.get("UnitTradeValue", 0))
                    })
                except Exception as e:
                    logger.error(f"Failed to parse ticker {ticker_name}: {e}")
            if ticker_batch:
                await self.db_manager.update_tickers_batch(ticker_batch)
        except Exception as e:
            logger.error(f"Failed to fetch ticker data: {e}")

    async def handle_get_balance(self):
        """
        FIX: The API returns { 'BTC': { 'Free': 0.45, 'Lock': 0.55 }, ... }
        We store only Free balances as a flat dict for use in the allocator.
        """
        try:
            response = await self.get_balance()
            raw_wallet = response.get("SpotWallet", {}) #No need for the rest, checked before in _request
            self.balance = {
                asset: float(data["Free"])
                for asset, data in raw_wallet.items()
            }
            logger.info(f"Balance synced: { {k: v for k, v in self.balance.items() if v > 0} }")
        except Exception as e:
            logger.error(f"Failed to get balance: {e}")

    async def update_order_data(self, order_detail: Dict):
        """Passes the order dictionary directly to the SQLite manager."""
        try:
            await self.db_manager.save_order(order_detail)
            logger.info(f"Order {order_detail.get('OrderID')} saved (Status: {order_detail.get('Status')})")
        except Exception as e:
            logger.error(f"Failed to save order to DB: {e}")

    async def handle_place_order(
        self, 
        symbol: str, 
        side: str, 
        quantity: float, 
        price: Optional[float] = None
    ):
        try:
            response = await self.place_order(symbol, side, quantity, price)
            order_detail = response.get("OrderDetail")
            if order_detail:
                await self.update_order_data(order_detail)
            else:
                logger.warning(f"place_order response had no OrderDetail: {response}")
        except Exception as e:
            logger.error(f"Failed to place order {side} {symbol}: {e}")

    async def handle_cancel_order(self, order_id: Optional[int] = None, pair: Optional[str] = None):
        try:
            response = await self.cancel_order(order_id, pair)
            for oid in response.get("CanceledList", []):
                await self.db_manager.cancel_order_by_id(oid)
        except Exception as e:
            logger.error(f"Failed to process cancellation: {e}")

    async def handle_query_order(
        self, 
        order_id=None, pair=None, pending_only=None, offset=None, limit=None
    ):
        res = await self.query_order(
            order_id=order_id, pair=pair, pending_only=pending_only,
            offset=offset, limit=limit
        )
        for order_detail in res:
            try:
                await self.update_order_data(order_detail)
            except Exception as e:
                logger.error(f"Failed to update order {order_detail}: {e}")