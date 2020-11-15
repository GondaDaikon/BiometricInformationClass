import numpy
import imageio
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

# run the network backwards, given a label, see what image it produces

# label to test
label = 6
# create the output signals for this label
targets = numpy.zeros(output_nodes) + 0.01
# all_values[0] is the target label for this record
targets[label] = 0.99
print(targets)

# get image data
image_data = n.backquery(targets)

# plot image data
image_data = image_data.reshape(28,28)
image_name = str(label) + '_backquery.png'
image_path = 'my_own_images/backquery_images/' + image_name
imageio.imwrite(image_path, image_data[:, :])