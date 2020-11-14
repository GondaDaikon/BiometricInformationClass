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

