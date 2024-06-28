from flask import Flask, request, jsonify
import joblib

model = joblib.load('Defaulter_prediction.joblib.gz')


app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predictions():
    data = request.get_json()

    result = model.predict(data['input_data'])

    response = {'prediction': result.tolist()}

    return jsonify(response)



if __name__ == '__main__':
    app.run(debug = True)