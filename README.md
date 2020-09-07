# classifier_HR
***
***

* HRのサンプルデータから退職者を予測するAutoMLの開発
* 上記で学習したモデルを利用し、予測値を返すAPIの作成

　　
## 機能一覧
***
* 退職者予測のAutoML
    * 2値の分類(Classification)タスクを扱う
    * カテゴリカル変数を指定し、one-hotエンコードを実行
    * モデル用データに施したのと同一データ前処理をスコア用データに対しても適用
    * 複数アルゴリズムからベストモデルを選択
    * 学習済みモデルを保存
    * 各モデルのスコアと性能評価指標をログ出力
    * 特徴量の重要度をcsv化
    * 学習済みモデルを呼び出しスコア用データに対して予測確率を付与。その結果をcsv化
* 学習済みモデルから予測値を返すAPI

　　
## ディレクトリ構成
***
ディレクトリ構成は下記の通り
```
classifier_HR/
        ┣ bin/        # 実行ファイルの配置場所
        ┣ data/       # 退職者予測のための訓練用データと検証用データの配置場所
        ┣ lib/        # ライブラリの配置場所
        ┣ log/        # ログ出力先
        ┣ model/      # 学習済みモデルの出力先
        ┗ api/        # 学習済みモデルを使い予測値を返すAPI
```

　　
## 実行方法
***
実行方法は下記の通り  
※python3環境が前提  

### 退職者予測AutoML
```
cd classifier_HR/bin
python classifier_HR.py
```

　　
### 学習済みモデルから予測値を返すAPI
ターミナルを２つ起動し、それぞれで以下を実行
```
cd classifier_HR/api
python hr_pred_api.py
```
```
cd classifier_HR/api
python hr_pred_api_test.py
```
