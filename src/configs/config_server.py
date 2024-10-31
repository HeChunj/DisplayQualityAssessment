from flask import Flask, request, jsonify

from config import ConfigManager

app = Flask(__name__)

config_manager = ConfigManager(config_file='config.yaml')
# db_host = config_manager.get('database', 'host')
# print(f"Database host: {db_host}")

# config_manager.set('app', 'debug', False)
# config_manager.save()


@app.route('/config/get_config', methods=['POST'])
def get_config():
    data = request.json
    config = config_manager.get(data['model'])
    return jsonify(config)


@app.route('/config/save_config', methods=['POST'])
def save_config():
    data = request.json
    print(data)
    for key, value in data['param'].items():
        config_manager.set(data['model'], key, value)
    config_manager.save()
    # config = config_manager.get(data['model'])
    return jsonify(data)


if __name__ == '__main__':
    config_server_name = "config_server"
    server = config_manager.get(config_server_name)
    app.run(server['host'], server['port'])
