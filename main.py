import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)
        self.synaptic_weights = 2 * np.random.random((3, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):
        for i in range(training_iterations):
            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):
        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
        return output

    def classification(self, df, ctype):
        if ctype == 'unclassified':
            classify = pd.DataFrame({'Answer_1': df['Answer_1'], 'Answer_2': df['Answer_2'], 'Answer_3': df['Answer_3'],
                                     'Answer_4': df['Answer_4'], 'Answer_5': df['Answer_5'], 'Answer_6': df['Answer_6'],
                                     'Answer_7': df['Answer_7'], 'Answer_8': df['Answer_8'], 'Answer_9': df['Answer_9'],
                                     'Answer_10': df['Answer_10'], 'Summary': df['Summary']})
            classify['Summary'] = classify['Answer_1'] + ' ' + classify['Answer_2'] + ' ' + classify['Answer_3'] + ' ' + classify['Answer_4'] + ' ' + classify['Answer_5'] + ' ' + classify['Answer_6'] + ' ' + classify['Answer_7'] + ' ' + classify['Answer_8'] + ' ' + classify['Answer_9'] + ' ' + classify['Answer_10']
            return classify

        elif ctype == 'classified':
            classify = pd.DataFrame({'Answer_1': df['Answer_1'], 'Answer_2': df['Answer_2'], 'Answer_3': df['Answer_3'],
                                     'Answer_4': df['Answer_4'], 'Answer_5': df['Answer_5'], 'Answer_6': df['Answer_6'],
                                     'Answer_7': df['Answer_7'], 'Answer_8': df['Answer_8'], 'Answer_9': df['Answer_9'],
                                     'Answer_10': df['Answer_10'], 'Summary': df['Summary'], 'Result': df['Result']})
            classify['Summary'] = classify['Answer_1'] + ' ' + classify['Answer_2'] + ' ' + classify['Answer_3'] + ' ' + classify['Answer_4'] + ' ' + classify['Answer_5'] + ' ' + classify['Answer_6'] + ' ' + classify['Answer_7'] + ' ' + classify['Answer_8'] + ' ' + classify['Answer_9'] + ' ' + classify['Answer_10']
            return classify

    def metrics(self, test_labels, predictions):
        accuracy = accuracy_score(test_labels, predictions)
        macro_precision = precision_score(test_labels, predictions, average='macro')
        macro_recall = recall_score(test_labels, predictions, average='macro')
        macro_f1 = f1_score(test_labels, predictions, average='macro')
        micro_precision = precision_score(test_labels, predictions, average='micro')
        micro_recall = recall_score(test_labels, predictions, average='micro')
        micro_f1 = f1_score(test_labels, predictions, average='micro')
        ceLoss = hamming_loss(test_labels, predictions)
        print("SVC Model Metrics ")
        print("Accuracy: {:.4f}\nHamming Loss: {:.4f}\nPrecision:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nRecall:\n  - Macro: {:.4f}\n  - Micro: {:.4f}\nF1-measure:\n  - Macro: {:.4f}\n  - Micro: {:.4f}" \
        .format(accuracy, ceLoss, macro_precision, micro_precision, macro_recall, micro_recall, macro_f1, micro_f1))

        return accuracy

    def give_prediction(self, answer, data):
        df = pd.read_json(data)

        classify = self.classification(df, 'classified')

        # Define X and y
        x = classify['Summary'].astype(str)
        y = classify['Result']

        # Split data 80% for training 20% for testing.
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)
        tfidf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', stop_words='english')
        x_train = tfidf.fit_transform(x_train)
        x = tfidf.transform(x)
        x_test = tfidf.transform(x_test)
        train_names = tfidf.get_feature_names_out()

        # Printing the tf-idf weights for n-grams in a random example.
        scores = pd.DataFrame()
        doc = 0
        feature_index = x_train[doc, :].nonzero()[1]
        tfidf_scores = zip(feature_index, [x_train[doc, index] for index in feature_index])

        # Tuples for scores and words in the randomly selected example
        tuples = [(train_names[index], score) for (index, score) in tfidf_scores]
        scores = scores.from_records(tuples, columns=['Words', 'Scores'])

        # scores in order
        scores = scores.sort_values(by=['Scores'], ascending=False)
        model = LinearSVC().fit(x_train, y_train)

        quiz = pd.DataFrame(answer, index=[0])
        unclassified = self.classification(quiz, 'unclassified')
        x_new = tfidf.transform(unclassified['Summary'])
        unclassified['Result'] = model.predict(x_new)
        unclassified.dropna()

        return unclassified



if __name__ == '__main__':
    neural_network = NeuralNetwork()
    json = {
        "Answer_1": "A organized person",
        "Answer_2": "Observations",
        "Answer_3": "Diagram",
        "Answer_4": "Conferences",
        "Answer_5": "I am able to retain what people have told me for some time.",
        "Answer_6": "I have a good imagination",
        "Answer_7": "Reading a book",
        "Answer_8": "Tables",
        "Answer_9": "I agree",
        "Answer_10": "Coordinate work for others",
        "Summary": "."
    }
    """print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_outputs = np.array([[0, 1, 1, 0]]).T

    neural_network.train(training_inputs, training_outputs, 10000)
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    A = str(input("Input 1: "))
    B = str(input("Input 2: "))
    C = str(input("Input 3: "))
    print("New situation: input data = ", A, B, C)
    print("Output data: ")
    print(neural_network.think(np.array([A, B, C])))"""

    df = pd.read_csv('classified_quiz_examples.csv', header=None)
    df.columns = ['Answer_1', 'Answer_2', 'Answer_3', 'Answer_4', 'Answer_5', 'Answer_6', 'Answer_7', 'Answer_8',
                  'Answer_9', 'Answer_10', 'Summary', 'Result']
    classify = neural_network.classification(df, 'classified')
    print(classify.head())

    # Define X and y
    x = classify['Summary'].astype(str)
    y = classify['Result']

    # Split data 80% for training 20% for testing.
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, shuffle=False)

    tfidf = TfidfVectorizer(smooth_idf=False, sublinear_tf=False, norm=None, analyzer='word', stop_words='english')
    x_train = tfidf.fit_transform(x_train)
    x = tfidf.transform(x)
    x_test = tfidf.transform(x_test)
    train_names = tfidf.get_feature_names_out()

    # Printing the tf-idf weights for n-grams in a random example.
    scores = pd.DataFrame()
    doc = 0
    feature_index = x_train[doc, :].nonzero()[1]
    tfidf_scores = zip(feature_index, [x_train[doc, index] for index in feature_index])

    # Tuples for scores and words in the randomly selected example
    tuples = [(train_names[index], score) for (index, score) in tfidf_scores]
    scores = scores.from_records(tuples, columns=['Words', 'Scores'])

    # scores in order
    scores = scores.sort_values(by=['Scores'], ascending=False)
    print(scores)

    model = LinearSVC().fit(x_train, y_train)
    predictions = model.predict(x_test)
    neural_network.metrics(y_test, predictions)

    quiz = pd.read_csv('unclassified_quiz_examples.csv', header=None)
    quiz.columns = ['Answer_1', 'Answer_2', 'Answer_3', 'Answer_4', 'Answer_5', 'Answer_6', 'Answer_7', 'Answer_8', 'Answer_9', 'Answer_10','Summary']
    unclassified = neural_network.classification(quiz, 'unclassified')
    print(unclassified.head())

    x_new = tfidf.transform(unclassified['Summary'])
    unclassified.shape
    unclassified['Result'] = model.predict(x_new)
    unclassified.dropna()
    print(unclassified.head())

    print(neural_network.give_prediction(json))






