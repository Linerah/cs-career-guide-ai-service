from flask import Flask, request, jsonify

app = Flask(__name__)


@app.route("/quizAI", methods=['POST'])
def quizAI():
    if request.method == 'POST':

        quizResults = request.json['answers']
        print(quizResults)
        return jsonify({'Success': '200'}, 200)
