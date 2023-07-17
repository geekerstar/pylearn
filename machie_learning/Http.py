from flask import Flask, request, jsonify

import machie_learning1

app = Flask(__name__)


@app.route('/api/get_user_info', methods=['GET'])
def get_user_info():
    user_id = request.args.get('user_id')
    # TODO: 查询数据库获取用户信息
    user_info = {'user_id': user_id, 'name': '张三', 'age': 25}
    return jsonify(user_info)


@app.route('/api/datasets_demo', methods=['GET'])
def datasets_demo():
    return machie_learning1.datasets_demo()


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
