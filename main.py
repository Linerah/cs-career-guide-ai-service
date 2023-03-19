import numpy as np
import pandas as pd


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
                                     'Answer_10': df['Answer_10']})
            classify['Answer_1'] = classify['Answer_1'] + ' ' + classify['Answer_2'] + ' ' + classify['Answer_3'] + ' ' + classify['Answer_4'] + ' ' + classify['Answer_5'] + ' ' + classify['Answer_6'] + ' ' + classify['Answer_7'] + ' ' + classify['Answer_8'] + ' ' + classify['Answer_9'] + ' ' + classify['Answer_10']
            return classify

        elif ctype == 'classified':
            classify = pd.DataFrame({'Answer_1': df['Answer_1'], 'Answer_2': df['Answer_2'], 'Answer_3': df['Answer_3'],
                                     'Answer_4': df['Answer_4'], 'Answer_5': df['Answer_5'], 'Answer_6': df['Answer_6'],
                                     'Answer_7': df['Answer_7'], 'Answer_8': df['Answer_8'], 'Answer_9': df['Answer_9'],
                                     'Answer_10': df['Answer_10'], 'Result': df['Result']})
            classify['Answer_1'] = classify['Answer_1'] + ' ' + classify['Answer_2'] + ' ' + classify['Answer_3'] + ' ' + classify['Answer_4'] + ' ' + classify['Answer_5'] + ' ' + classify['Answer_6'] + ' ' + classify['Answer_7'] + ' ' + classify['Answer_8'] + ' ' + classify['Answer_9'] + ' ' + classify['Answer_10']
            return classify


if __name__ == '__main__':
    neural_network = NeuralNetwork()
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
                  'Answer_9', 'Answer_10', 'Result']
    classify = neural_network.classification(df, 'classified')
    print(classify.head())



