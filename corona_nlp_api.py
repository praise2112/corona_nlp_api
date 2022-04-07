import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS
from util import predict_sentiment_onnx

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
CORS(app)


@app.route('/')
def check_API():
  return 'API is up'


@app.route('/sentiment/<path:text>', methods=['GET'])
def predict_sentiment(text):
  try:
    result = predict_sentiment_onnx(text)
    return jsonify({"success": True,  "prediction": result})
  except Exception as e:
    print(e)
    return jsonify({"success": False,  "info": "Something went wrong"})


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--port', type=int, nargs='?', const=1, help='sever port', default=80)
  args = parser.parse_args()
  app.run(host='0.0.0.0', debug=True, port=args.port)
