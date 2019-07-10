import numpy as np
from scipy.special import expit, logit


class UndefinedError(Exception):
    def __init__(self):
        super().__init__('Function is not defined at 0.0 and 1.0')


class NeuralNetwork:
    def __init__(self, in_nodes, hidden_nodes, out_nodes, learning_grate, activation=expit,
                 inverse_activation=logit):
        '''
        :param in_nodes: Количество входных вершин
        :param hidden_nodes: Количество скрытых вершин
        :param out_nodes: Количество выходных вершин
        :param learning_grate: Коэффициент обучения
        :param activation: Функция активации нейрона
        '''
        self.inodes = in_nodes + 1
        self.hnodes = hidden_nodes
        self.onodes = out_nodes

        self.lrate = learning_grate

        self.wih = np.random.normal(0.0, pow(self.inodes, -0.5), (self.hnodes, self.inodes))
        self.who = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.onodes, self.hnodes))

        self.af = activation
        self.af_inv = inverse_activation

    def train(self, inputs_list, targets_list):
        # Прогоняем входные данные через сеть
        inputs = np.array(np.hstack(([1], inputs_list)), dtype=float, ndmin=2).T

        # Считаем вход для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)

        # Применяем функцию активации
        hidden_outputs = self.af(hidden_inputs)

        # Аналогично для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.af(final_inputs)

        # Транспонируем лист с ответами
        targets = np.array(targets_list, dtype=float, ndmin=2).T

        output_errors = targets - final_outputs
        hidden_errors = np.dot(self.who.T, output_errors)
        self.who += self.lrate * np.dot(output_errors * final_outputs * (1.0 - final_outputs), hidden_outputs.T)
        self.wih += self.lrate * np.dot(hidden_errors * hidden_outputs * (1.0 - hidden_outputs), inputs.T)

    def query(self, inputs_list):
        # Транспонируем исходную матрицу
        inputs = np.array(np.hstack(([1], inputs_list)), dtype=float, ndmin=2).T

        # Считаем вход для скрытого слоя
        hidden_inputs = np.dot(self.wih, inputs)

        # Применяем функцию активации
        hidden_outputs = self.af(hidden_inputs)

        # Аналогично для выходного слоя
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.af(final_inputs)

        return final_outputs

    def backward_query(self, output_list):
        if 1 in output_list:
            raise UndefinedError
        final_outputs = np.array(output_list, dtype=float, ndmin=2)
        final_inputs = self.af_inv(final_outputs)
        hidden_outputs = final_inputs.dot(self.who)

        # Приводим значения к области значений сигмоиды
        hidden_outputs -= np.min(hidden_outputs)
        hidden_outputs /= np.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01

        hidden_inputs = self.af_inv(hidden_outputs)
        inputs = hidden_inputs.dot(self.wih)

        # Снова приводим
        inputs -= np.min(inputs)
        inputs /= np.max(inputs)
        inputs *= 0.98
        inputs += 0.01

        return inputs[:, 1:]


if __name__ == '__main__':
    n = NeuralNetwork(2, 5, 1, learning_grate=1)

    Binary_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

    OR_outputs = np.array([[0], [1], [1], [1]])
    AND_outputs = np.array([[0], [0], [0], [1]])
    XOR_outputs = np.array([[0], [1], [1], [0]])
    NOR_outputs = np.array([[1], [0], [0], [0]])
    SELECTX1_outputs = np.array([[0], [0], [1], [1]])
    SELECTX2_outputs = np.array([[0], [1], [0], [1]])

    epochs = 1000
    for i in range(epochs):
        for sample, ans in zip(Binary_inputs, XOR_outputs):
            n.train(sample, ans)

    for sample in Binary_inputs:
        print(n.query(sample))
