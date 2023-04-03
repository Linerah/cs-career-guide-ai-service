from flask import Flask, request, jsonify
from main import NeuralNetwork
app = Flask(__name__)


@app.route("/quizAI", methods=['POST'])
def quizAI():
    if request.method == 'POST':
        quizResults = request.json['answers']

        neural_network = NeuralNetwork()
        prediction = neural_network.give_prediction(quizResults)
        print(prediction)
        return jsonify({'result': prediction.loc[0, 'Result']})
