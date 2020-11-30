import numpy
import scipy.special

# ニューラルネットワーククラスの定義
class neuralNetwork:

    # ニューラルネットワークの初期化
    def __init__(self, inputnodes, hiddennods, outputnods, learningrate, hiddenlayers=1):
        # 入力層、隠れ層、出力層のノード数の設定
        self.inodes = inputnodes
        self.hnodes = hiddennods
        self.onodes = outputnods
        # 隠れ層の数
        self.hlayers = hiddenlayers

        # リンクの重み行列 wih と who
        # 行列内の重み w_i_j, ノードiから次の層のノードjへのリンクの重み
        # w11 w21
        # w12 w22 など
        # numpy.random.normal(平均、標準偏差、（行、列）)
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.inodes))

        # -- 隠れ層の重み生成 開始 --
        self.whh = numpy.empty((self.hlayers-1,self.hnodes,self.hnodes))

        for layer in range(self.hlayers-1):
            tmp = numpy.random.normal(0.0, pow(self.hnodes, -0.5),(self.hnodes, self.hnodes))
            self.whh[layer,:,:] = tmp
        # -- 隠れ層の重み生成 終了 --

        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5),(self.onodes, self.hnodes))

        # 学習率の設定
        self.lrate = learningrate

        # 誤差の設定
        self.loss = 0

        # 活性化関数はシグモイド関数
        self.activation_function = lambda x: scipy.special.expit(x)
        self.inverse_activation_function = lambda x: scipy.special.logit(x)

        pass

    # ニューラルネットワークの学習
    def train(self, inputs_list, targets_list):
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        firsthlayer_input = numpy.dot(self.wih, inputs)
        # 隠れ層で結合された信号を活性化関数により出力
        firsthlayer_output = self.activation_function(firsthlayer_input)

        ## -- 隠れ層間の計算 開始 --
        lasthlayer_output = firsthlayer_output
        hidden_inputs = numpy.empty((self.hlayers,self.hnodes,1))
        hidden_outputs = numpy.empty((self.hlayers,self.hnodes,1))
        hidden_inputs[0,:,:] = firsthlayer_input
        hidden_outputs[0,:,:] = firsthlayer_output
        for i, whh in enumerate(self.whh):
            hidden_inputs[i+1,:,:] = numpy.dot(whh, hidden_outputs[i,:,:])
            hidden_outputs[i+1,:,:] = self.activation_function(hidden_inputs[i+1,:,:])
            if i == (len(self.whh)-1):
                lasthlayer_output = hidden_outputs[i+1,:,:]
        hidden_outputs = numpy.flipud(hidden_outputs)
        ## -- 隠れ層間の計算 終了 --

        # 出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.who, lasthlayer_output)
        # 出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)

        # 出力層の誤差　＝　（目標出力　ー　最終出力）
        output_errors = targets - final_outputs

        # 出力層の誤差を保存
        eroor = 0
        for errror in output_errors:
                eroor += errror*errror
        self.loss += eroor/len(output_errors)

        # 隠れ層の誤差は出力層の誤差をリンクの重みの割合で分配
        lasthlayer_errors = numpy.dot(self.who.T, output_errors)

        ## -- 隠れ層間の誤差分配 開始 --
        hidden_errors = numpy.empty((self.hlayers,self.hnodes,1))
        hidden_errors[0,:,:] = lasthlayer_errors
        firsthlayer_errors = lasthlayer_errors
        for i, whh in enumerate(reversed(self.whh)):
            hidden_errors[i+1,:,:] = numpy.dot(whh.T, hidden_errors[i,:,:])
            if i == (len(self.whh)-1):
                firsthlayer_errors = hidden_errors[i+1,:,:]
        ## -- 隠れ層間の誤差分配 終了 --

        # 隠れ層と出力層の間のリンクの重みを更新
        self.who += self.lrate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(lasthlayer_output))
        
        ## -- 隠れ層間の重みを更新 開始 --
        for i, whh in enumerate(reversed(self.whh)):
            self.whh[(len(self.whh)-1) -i,:,:] += self.lrate * numpy.dot(\
                (hidden_errors[i,:,:] * hidden_outputs[i,:,:] * (1.0 - hidden_outputs[i,:,:])),\
                    numpy.transpose(hidden_outputs[i+1,:,:]))
        ## -- 隠れ層間の重みを更新 終了 --

        # 入力層と隠れ層の間のリンクの重みを更新
        self.wih += self.lrate * numpy.dot((firsthlayer_errors * firsthlayer_output * (1.0 - firsthlayer_output)),numpy.transpose(inputs))

        pass
    
    # ニューラルネットワークへの照会
    def query(self, inputs_list):
        # 入力リストを行列に変換
        inputs = numpy.array(inputs_list, ndmin=2).T

        # 隠れ層に入ってくる信号の計算
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 隠れ層で結合された信号を活性化関数により出力
        hiden_outputs = self.activation_function(hidden_inputs)

        # -- 隠れ層間の信号の計算 開始 --
        for whh in self.whh:
            hidden_inputs = numpy.dot(whh, hiden_outputs)
            hiden_outputs = self.activation_function(hidden_inputs)
        # -- 隠れ層間の信号の計算 終了 --

        # 出力層に入ってくる信号の計算
        final_inputs = numpy.dot(self.who, hiden_outputs)
        # 出力層で結合された信号を活性化関数により出力
        final_outputs = self.activation_function(final_inputs)

        return final_outputs

    def backquery(self, targets_list):
        # transpose the targets list to a vertical array
        final_outputs = numpy.array(targets_list, ndmin=2).T
        
        # calculate the signal into the final output layer
        final_inputs = self.inverse_activation_function(final_outputs)

        # calculate the signal out of the hidden layer
        hidden_outputs = numpy.dot(self.who.T, final_inputs)
        # scale them back to 0.01 to .99
        hidden_outputs -= numpy.min(hidden_outputs)
        hidden_outputs /= numpy.max(hidden_outputs)
        hidden_outputs *= 0.98
        hidden_outputs += 0.01
        
        # calculate the signal into the hidden layer
        hidden_inputs = self.inverse_activation_function(hidden_outputs)
        
        # calculate the signal out of the input layer
        inputs = numpy.dot(self.wih.T, hidden_inputs)
        # scale them back to 0.01 to .99
        inputs -= numpy.min(inputs)
        inputs /= numpy.max(inputs)
        inputs *= 0.98
        inputs += 0.01
        
        return inputs
    
    def save_parameters(self):
        parameters = numpy.array([self.inodes, self.hnodes, self.onodes, self.lrate, self.hlayers])
        numpy.save('trainedModel/parameters.npy',parameters)
        self.save_weights()

    def save_weights(self):
        numpy.save('trainedModel/weights/wih.npy', self.wih)
        numpy.save('trainedModel/weights/whh.npy', self.whh)
        numpy.save('trainedModel/weights/who.npy', self.who)

    def load_weights(self):
        self.wih = numpy.load('trainedModel/weights/wih.npy')
        self.whh = numpy.load('trainedModel/weights/whh.npy')
        self.who = numpy.load('trainedModel/weights/who.npy')
