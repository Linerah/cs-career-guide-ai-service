import json
from flask import Flask, request, jsonify
from main import NeuralNetwork
import pymongo
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)

app.secret_key = b'\xcc^\x91\xea\x17-\xd0W\x03\xa7\xf8J0\xac8\xc5'

app.config['MONGO_DBNAME'] = 'user_auth'

client = pymongo.MongoClient(
    "mongodb+srv://admin:NtXLrfmOBLhl00bm@capstoneauth.25mmcqj.mongodb.net/?retryWrites=true&w=majority", tls=True,
    tlsAllowInvalidCertificates=True)
db = client['user-auth']

"""@app.route('/adddata', methods=['GET'])
@cross_origin()
def add_data():
    data = []
    with open('classified_quiz_examples.csv') as file:
        filereader = csv.DictReader(file)
        for row in filereader:
            data.append(row)
    db.training.insert_many(data)
    return jsonify(data)"""


@app.route("/quizAI", methods=['POST', 'GET'])
@cross_origin()
def quizAI():
    if request.method == 'POST':

        quizResults = request.json['results']['answers']
        isProfessor = request.json['currentUser']['isProfessor']

        if isProfessor:
            db.training.insert_one(quizResults)
            return jsonify({'Status': 'OK'})
        else:
            projection = {"_id": 0}
            cursor = db.training.find({}, projection)

            cursorList = list(cursor)

            savedData = json.dumps(cursorList)
            neural_network = NeuralNetwork()
            prediction = neural_network.give_prediction(quizResults, savedData)

            return jsonify({'result': prediction.loc[0, 'Result']})
