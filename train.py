import numpy
import scipy.ndimage
from tqdm import tqdm
from neuralnetwork import neuralNetwork

# 入力層、隠れ層、出力層のノード数
input_nodes = 784
hidden_nodes = 100
output_nodes = 10

hiddenlayers = 3

# 学習率　＝　0.3
learning_rate = 0.1
# ニューラルネットワークのインスタンス生成
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, hiddenlayers)

# MNIST 訓練データのCSV ファイルを読み込んでリストにする
training_data_file = open("dataSet/mnist_train.csv", 'r')
training_data_list = training_data_file.readlines()
training_data_file.close()

epochs = 1

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

        ## create rotated variations
        # rotated anticlockwise by x degrees
        inputs_plusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        n.train(inputs_plusx_img.reshape(784), targets)
        # rotated clockwise by x degrees
        inputs_minusx_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        n.train(inputs_minusx_img.reshape(784), targets)
        
        # rotated anticlockwise by 10 degrees
        #inputs_plus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), 10, cval=0.01, order=1, reshape=False)
        #n.train(inputs_plus10_img.reshape(784), targets)
        # rotated clockwise by 10 degrees
        #inputs_minus10_img = scipy.ndimage.interpolation.rotate(inputs.reshape(28,28), -10, cval=0.01, order=1, reshape=False)
        #n.train(inputs_minus10_img.reshape(784), targets)

        #progress training_bar
        training_bar.update(1)
        pass
    pass
training_bar.close()

n.save_parameters()