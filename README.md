# classifier_HR
***
***

当リポジトリは退職予測用MLの実行環境

　　
## Function
***
* 2値の分類(Classification)タスクを扱う
* カテゴリカル変数を指定し、one-hotエンコードを実行
* モデル用データに施したのと同一データ前処理をスコア用データに対しても適用
* 特徴量の重要度をcsv化
* 複数アルゴリズムのモデル構築
* 学習済みモデルを保存
* 各モデルのスコアと性能評価指標をログ出力

　　
## Directory
***
ディレクトリ構成は下記の通り
```
classifier_HR/
        ┣ bin/        # 実行ファイルの配置場所
        ┣ data/       # 退職予測のための訓練用データと検証用データの配置場所
        ┣ lib/        # ライブラリの配置場所
        ┣ log/        # ログ出力先
        ┗ model/      # 学習済みモデルの出力先
```

　　
## Usage
***
実行方法は下記の通り
```
cd classifier_HR/bin
python classifier_HR.py
```
※python3環境が前提
