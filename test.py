import numpy
from tqdm import tqdm
from neuralnetwork import neuralNetwork

# ===学習モデルを読み込み開始====
parameters = numpy.load('trainedModel/parameters.npy')
# 入力層、隠れ層、出力層のノード数
input_nodes = int(parameters[0])
hidden_nodes = int(parameters[1])
output_nodes = int(parameters[2])
# 学習率
learning_rate = parameters[3]
# ニューラルネットワークのインスタンス生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 保存した重みを読み込む
n.load_weights()
# ===学習モデルを読み込み終わり====

# MNIST 訓練データのCSV ファイルを読み込んでリストにする
test_data_file = open("dataSet/mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

scorecard = []

record_count = len(test_data_list)
recording_bar = tqdm(total = record_count)
recording_bar.set_description('now recording...')

for record in test_data_list:
    all_values = record.split(',')
    correct_label = int(float(all_values[0]))
    inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
    outputs = n.query(inputs)
    label = numpy.argmax(outputs)
    if (label == correct_label):
        scorecard.append(1)
    else:
        scorecard.append(0)
        pass
    recording_bar.update(1)
    pass
recording_bar.close()

scorecard_array = numpy.asarray(scorecard)
print("parformance = ", scorecard_array.sum() / scorecard_array.size)