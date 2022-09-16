import flask
import logging
import traceback
from flask_cors import *
import W2NER
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)

app = flask.Flask(__name__)
CORS(app, supports_credentials=True)

w2ner = W2NER.W2NER()

@app.route("/ner", methods=['POST'])
def ner():
    result = {
        'message': u'失败',
        'status':5201
    }
    if flask.request.method == 'POST':
        if flask.request.is_json:
            json_data = flask.request.get_json()
            try:
                sentence = json_data.get('sentence')
                res = w2ner.predict(list(sentence))
                result['ner'] = res
                result['message'] = u'成功'
                result['status'] = 5200
                return flask.jsonify(result)
            except Exception as e:
                result['message'] = u'解析失败'
                result['status'] = 5202
                logger.info(traceback.format_exc())
                return flask.jsonify(result)
        else:
            result['message'] = u'无法解析'
            result['status'] = 5203

if __name__ == '__main__':
    print('server starting....')
    app.run(
        host='0.0.0.0',
        port=config.port
    )
