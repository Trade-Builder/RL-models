from websocket_server import WebsocketServer
from quantylab.rltrader.deployer import ModelDeployer
from quantylab.rltrader.data_manager import load_crypto_data, preprocess_crypto_data, COLUMNS_CRYPTO_DATA
import json
import traceback

# Initialize deployer
deployer = ModelDeployer(model_name='20251027124731', model_dir='models/best')
is_initialized = False

def init_model_single_tf(data):
    """Initialize model with single timeframe data
    
    Supports two formats:
    1. Simple (close only): [close1, close2, ..., close200]
    2. OHLCV (dict): {
        'close': [...],888888888888
        'open': [...],   # optional
        'high': [...],   # optional
        'low': [...],    # optional
        'volume': [...]  # optional
    }
    """
    global is_initialized
    
    # Check if data is dict (OHLCV) or list (close only)
    if isinstance(data, dict):
        # OHLCV 형식
        closes = data.get('close')
        opens = data.get('open')
        highs = data.get('high')
        lows = data.get('low')
        volumes = data.get('volume')
        
        deployer.load_initial_ohlcv(closes, opens, highs, lows, volumes)
    else:
        # 단순 close 리스트
        deployer.load_initial_closes(data)
    
    is_initialized = True
    return {"status": "initialized", "method": "single_timeframe"}

def init_model_multi_tf(data, base_interval='minute1'):
    """Initialize model with multi-timeframe data
    
    Expected data format:
    {
        'minute1': {'close': [...], 'volume': [...]},
        'minute5': {'close': [...], 'volume': [...]},
        ...
    }
    """
    global is_initialized
    deployer.load_initial_multi_tf(data, base_interval=base_interval)
    is_initialized = True
    return {"status": "initialized", "method": "multi_timeframe", "base_interval": base_interval}

def run_model_single_tf(idx, close_price):
    """Run prediction with single close price"""
    global is_initialized
    assert is_initialized, "Model not initialized. Call 'init' first."
    
    action, conf, unit, pv = deployer.on_new_close(close_price, execute=False)
    action_str = {0:'BUY', 1:'SELL', 2:'HOLD'}.get(action, 'UNKNOWN')
    return {
        'index': idx, 
        'close': close_price, 
        'action': action_str, 
        'action_code': action,
        'confidence': conf, 
        'trade_unit': unit, 
        'portfolio_value': pv
    }

def run_model_multi_tf(idx, tf_data, base_interval='minute1'):
    """Run prediction with multi-timeframe data
    
    Expected tf_data format:
    {
        'minute1': {'close': 50000000.0, 'volume': 5.5},
        'minute5': {'close': 50100000.0, 'volume': 6.0},
        ...
    }
    """
    global is_initialized
    assert is_initialized, "Model not initialized. Call 'init' first."
    
    action, conf, unit, pv = deployer.on_new_multi_tf(tf_data, base_interval=base_interval, execute=False)
    action_str = {0:'BUY', 1:'SELL', 2:'HOLD'}.get(action, 'UNKNOWN')
    return {
        'index': idx,
        'action': action_str,
        'action_code': action,
        'confidence': conf,
        'trade_unit': unit,
        'portfolio_value': pv,
        'base_interval': base_interval
    }

def set_portfolio_state(num_stocks, balance, avg_buy_price=None):
    """Set portfolio state for simulation
    
    Args:
        num_stocks: Number of coins held (e.g., 0.5 BTC)
        balance: Cash balance (e.g., 50000000)
        avg_buy_price: Average buy price (optional, defaults to current price)
    
    Example:
    {
        "action": "set_portfolio",
        "num_stocks": 0.5,
        "balance": 50000000,
        "avg_buy_price": 65000000
    }
    """
    global is_initialized
    assert is_initialized, "Model not initialized. Call 'init' first."
    
    deployer.set_portfolio_state(
        num_stocks=float(num_stocks),
        balance=float(balance) if balance is not None else None,
        avg_buy_price=float(avg_buy_price) if avg_buy_price is not None else None
    )
    
    return {
        "status": "portfolio_set",
        "num_stocks": deployer.agent.num_stocks,
        "balance": deployer.agent.balance,
        "avg_buy_price": deployer.agent.avg_buy_price,
        "portfolio_value": deployer.agent.portfolio_value
    }

def handle_server(request):
    """Handle WebSocket requests
    
    Request formats:
    
    1. Initialize with single timeframe:
    {
        "action": "init",
        "method": "single",
        "data": [close1, close2, ..., close200]
    }
    
    2. Initialize with multi-timeframe:
    {
        "action": "init",
        "method": "multi",
        "base_interval": "minute1",
        "data": {
            "minute1": {"close": [...], "volume": [...]},
            "minute5": {"close": [...], "volume": [...]}
        }
    }
    
    3. Run prediction (single TF):
    {
        "action": "run",
        "method": "single",
        "index": 0,
        "data": 50000000.0
    }
    
    4. Run prediction (multi TF):
    {
        "action": "run",
        "method": "multi",
        "index": 0,
        "base_interval": "minute1",
        "data": {
            "minute1": {"close": 50000000.0, "volume": 5.5},
            "minute5": {"close": 50100000.0, "volume": 6.0}
        }
    }
    """
    action = request.get('action', '').lower()
    
    if action == 'init':
        method = request.get('method', 'single').lower()
        data = request.get('data')
        
        if method == 'single':
            return init_model_single_tf(data)
        elif method == 'multi':
            base_interval = request.get('base_interval', 'minute1')
            return init_model_multi_tf(data, base_interval)
        else:
            raise ValueError(f"Unknown init method: {method}")
    
    elif action == 'run':
        method = request.get('method', 'single').lower()
        idx = request.get('index', 0)
        data = request.get('data')
        
        if method == 'single':
            return {"status": "success", "result": run_model_single_tf(idx, data)}
        elif method == 'multi':
            base_interval = request.get('base_interval', 'minute1')
            return {"status": "success", "result": run_model_multi_tf(idx, data, base_interval)}
        else:
            raise ValueError(f"Unknown run method: {method}")
    
    elif action == 'set_portfolio':
        num_stocks = request.get('num_stocks', 0.0)
        balance = request.get('balance')
        avg_buy_price = request.get('avg_buy_price')
        return set_portfolio_state(num_stocks, balance, avg_buy_price)
    
    elif action == 'status':
        return {
            "status": "success",
            "initialized": is_initialized,
            "model_name": deployer.model_name
        }
    
    else:
        raise ValueError(f"Unknown action: {action}")

def run_server(handler):
    def message_received(client, server, message):
        print(f"메시지 수신: {message}")
        message = json.loads(message)
        messageId = message['messageId']
        response = handler(message)
        response['messageId'] = messageId
        response_text = json.dumps(response)
        server.send_message(client, response_text)

    server = WebsocketServer(host='127.0.0.1', port=5577)
    server.set_fn_message_received(message_received)

    print("웹소켓 서버 시작 (단일 클라이언트, 포트 5577)")
    server.run_forever()

if __name__ == "__main__":
    run_server(handler=handle_server)

