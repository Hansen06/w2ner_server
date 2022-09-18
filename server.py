import json

import flask
import logging
import traceback
from flask_cors import *
import W2NER
import config
from flask import Response

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = flask.Flask(__name__)
CORS(app, supports_credentials=True)

w2ner = W2NER.W2NER()


def creat_response(result_dict):
    result_str = json.dumps(result_dict, ensure_ascii=False)
    return Response(response=result_str, status=200, mimetype="application/json")


@app.route("/ner", methods=['POST'])
def ner():
    result = {
        'message': u'失败',
        'status': 5201
    }
    if flask.request.method == 'POST':
        if flask.request.is_json:
            json_data = flask.request.get_json()
            try:
                sentences = json_data.get('sentence')
                res = w2ner.predict(sentences)
                result['ner'] = res
                result['message'] = u'成功'
                result['status'] = 5200
                # return flask.jsonify(result)
                return creat_response(result)
            except Exception as e:
                result['message'] = u'解析失败'
                result['status'] = 5202
                logger.info(traceback.format_exc())
                # return flask.jsonify(result)
                return creat_response(result)
        else:
            result['message'] = u'无法解析'
            result['status'] = 5203
            return creat_response(result)


if __name__ == '__main__':
    print('server starting....')
    app.run(
        host='0.0.0.0',
        port=config.port
    )
