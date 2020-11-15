# BiometricInformationClass

"BiometricInformationClass"は手書き文字認識のサンプルプログラムです。  

## 実行手順  
### 1. 動作環境構築  

Pipfileのディレクトリで、以下のコマンド： 
``` 
$ pipenv install  
```

### 2. データセットのダウンロード  

・訓練データ：http://www.pjreddie.com/media/files/mnist_train.csv  

・テストデータ：http://www.pjreddie.com/media/files/mnist_test.csv  

```
data_root  
├── Pipfile  
├── Pipfile.lock  
├── README.md  
├── dataSet  
│   ├── mnist_test.csv  
│   └── mnist_train.csv  
├── neuralnetwork.py  
├── pylintrc  
└── train.py  
```
### 3. データセットの学習  
プログラム内で読み込まれた*.csvファイルを読み込み学習を開始します。  
```
$ python train.py  
```
### 4. 学習精度の計算  
trainModelに置かれたの学習済みのモデルを読み込んで学習精度(正答率)を計算します。  
```
$ python test.py  
```
### 5. 学習済みモデルのBack Query  
trainModelに置かれたの学習済みのモデルを読み込んで、プログラム内で設定したlabelBack Queryをします。  
Back Queryの結果は"dataSet/my_own_images/backquery_images"に保存されます。
```
$ python backquery.py  
```
