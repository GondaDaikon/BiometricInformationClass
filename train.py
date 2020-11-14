import numpy
from tqdm import tqdm
from neuralnetwork import neuralNetwork

# 入力層、隠れ層、出力層のノード数
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

# 学習率　＝　0.3
learning_rate = 0.3
# ニューラルネットワークのインスタンス生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# MNIST 訓練データのCSV ファイルを読み込んでリストにする
training_data_file = open("dataSet/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 2

training_count = len(training_data_list) * epochs
training_bar = tqdm(total = training_count)
training_bar.set_description('now training... ')

for e in range(epochs):
    # 訓練データの全データに対して実行
    for recode in training_data_list:
        # データを','でsplit
        all_values = recode.split(',')
        # 入力値のスケーリングとシフト
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        # 目標配列の生成（ラベル位置が0.99　残りは0.01）
        targets = numpy.zeros(output_nodes) + 0.01
        # all_values[0]はこのデータのラベル
        targets[int(all_values[0])] = 0.99
        n.train(inputs,targets)
        #progress training_bar
        training_bar.update(1)
        pass
    pass
training_bar.close()

# MNIST 訓練データのCSV ファイルを読み込んでリストにする
test_data_file = open("dataSet/my_own_images/my_own_dataset.csv", 'r')
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