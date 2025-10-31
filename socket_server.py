from websocket_server import WebsocketServer
from quantylab.rltrader.deployer import ModelDeployer
from quantylab.rltrader.data_manager import load_crypto_data, preprocess_crypto_data, COLUMNS_CRYPTO_DATA
import json

deployer = ModelDeployer(model_name='20251027124731')

def init_model(data):
    deployer.load_initial_closes(data)

def run_model(idx, data):
    action, conf, unit, pv = deployer.on_new_close(data, execute=False)
    action_str = {0:'BUY',1:'SELL',2:'HOLD'}.get(action, 'UNK')
    return {'index': idx, 'close': data, 'action': action_str, 'conf': conf, 'unit': unit, 'pv': pv}

def handle_server(data):
    if data['action'] == 'init':
        init_model(data['data'])
        return {"status": "initialized"}
    
    elif data['action'] == 'run':
        result = run_model(data['index'], data['data'])
        return {"status": "running", "result": result}
    
    return {"error": "unknown_action"}

def run_server(handler):
    def message_received(client, server, message):
        response = handler(json.loads(message))
        response_text = json.dumps(response)
        server.send_message(client, response_text)

    server = WebsocketServer(host='127.0.0.1', port=5577)
    server.set_fn_message_received(message_received)

    print("웹소켓 서버 시작 (단일 클라이언트, 포트 5577)")
    server.run_forever()

if __name__ == "__main__":
    run_server(handler=handle_server)

