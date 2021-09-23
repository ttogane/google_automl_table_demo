# Google Auto ML Table Demo
## 目的
昨今のMLブームでMLを使ったサービスが増えてきた中で、開発エンジニアでも簡単にMLを利用できるようにAuto ML(自動でパラメーターチューニング&アルゴリズム選定)を行ってくれるサービスが増えてきた。
そこで実際に[GCP AutoML Table Quick start](https://cloud.google.com/automl-tables/docs/quickstart)をやってみてAuto MLとML自体の理解を深める。　　
</br>

## 前提
そもそも`Auto ML`とは、AIの構築(≒ MLを用いた予測 or 分類モデルの作成)を行う上でデータサイエンティストが行うアルゴリズム選定とパラメータチューニングを**自動で**行ってくれるもの。  

今までエンジニアはAI構築の際に、研究部門やデータサイエンティストにモデルの作成を依頼して作ってもらうということが多かったが、`Auto ML`によってデータさえあればエンジニアでも最適なモデルを作成/利用することができるようになった。  

基本的にMLではデータを「ターゲット」と「特徴パラメータ」で定義することで学習を行う。  
  
> ターゲット: 予測/分類したい対象のデータ（学習時の答えになるもの= ラベル）  
> 特徴パラメータ: ターゲット以外のデータ(学習時の出題になるもの)

この関係は「予測モデル」も「分類モデル」も同じ関係にある。  
  (= 出題と回答だけをみて特徴に対する回答のパターンを覚えていくイメージ)
</br>

### example
例えばユーザーの「年収」を推測する場合の「ターゲット」と「特徴パラメータ」は以下になる。　　
| 特徴パラメータ | ターゲット |
| ------- | ------- |
| 年齢, 性別, 学歴, 職種, 居住地, 結婚歴 | 年収 |
</br>


## 1, 準備作業
### プロジェクト構成
```bash
google_automl_table_demo
├── README.md
├── dataset
│   └── bank-marketing_csv.csv
├── out
│   └── feature_importance.csv
└── screen_shots
    ├── 01_create_new_dataset.png
    ├── 02_import_dataset_csv_from_pc.png
    ├── 03_create_dataset_is_done.png
    ├── 04_training_settings.png
    ├── 05_model_evaluation.png
    └── 06_feature_importance.png
```
今回使うデータセットは`dataset/bank-marketing_csv.csv`で、これはオープンソースの[Bank Marketing](https://datahub.io/machine-learning/bank-marketing)で、
列はわかりやすいように改定した。
  
なお、各列の構成は以下
| 改定前 | 改定後 | 説明 |
| -------- | -------- | -------- |
| v1 | age | customer age |
| v2 | job_type | type of job |
| v3 | marital | marital status |
| v4 | education | education |
| v5 | default | has credit in default? (binary: “yes”,“no”) |
| v6 | balance | average yearly balance, in euros |
| v7 | housing | has housing loan? (binary: “yes”,“no”) |
| v8 | loan | has personal loan? (binary: “yes”,“no”) |
| v9 | contact | contact communication type |
| v10 | day | last contact day of the month |
| v11 | month | last contact month of year |
| v12 | duration | last contact duration, in seconds |
| v13 | campaign | number of contacts performed during this campaign and for this client |
| v14 | pdays | number of days that passed by after the client was last contacted from a previous campaign |
| v15 | previous | number of contacts performed before this campaign and for this client |
| v16 | poutcome | outcome of the previous marketing campaign |
| v17 | is_deposit | has the client subscribed a term deposit? |

##  2, ハンズオン
**※詳しくは公式を参照**  
</br>

### STEP 1: データセットを作成する
「新しいデータセット」からデータセット作成画面へ  
<img width=600 height=300 alt="01_create_new_dataset" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/01_create_new_dataset.png"/>  
</br>

### STEP 2: データセットのインポート
1, 「パソコンからファイルをアップロード」を選択し、「アップロード先のフォルダを選択」で既存のバケットを選択するか新規で任意のバケットを作り選択する。  
<img width=600 height=300 alt="02_import_dataset_csv_from_pc" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/02_import_dataset_csv_from_pc.png"/>
</br>

2, 選択が完了したら「インポート」を実行する。  
</br>

### STEP 3: トレーニングを開始する
1, ターゲット列に`is_deposit`を選択し、「モデルトレーニング」を実行する。    
<img width=600 height=300 alt="03_create_dataset_is_done" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/03_create_dataset_is_done.png"/>
</br>

2, 設定項目が出てくるので、各種設定を行い「モデルトレーニング」を実行する。
<img width=600 height=300 alt="04_training_settings" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/04_training_settings.png"/>
</br>

### STEP 4: モデルの評価
このモデルでは、1 は悪い結果（この銀行に預金が行われないこと）を表し、2 は良い結果（この銀行に預金が行われること）を表します。  
<img width=600 height=300 alt="05_model_evaluation" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/05_model_evaluation.png"/>
</br>

**用語の意味**
例えば果物を分類することを考えた場合「りんご、バナナ、いちご」を分類したい。
この時、
```
Log Loss = 実際のラベルからどのくらい違っていたのか(0が一番いい)

precision(適合率) = (真陽性) / (真陽性 + 偽陽性)
                 = 正しい果物を推定した数 / (正しい果物を推定した数 + 間違った果物を推定した数)
                 = りんごと推定したサンプルの中で、実際に正解した割合

recall(再現率) = (真陽性) / (真陽性 + 偽陰性) 
              = 正しい果物を推定した数　/ (正しい果物を推定した数 + 間違って正しい果物でないと推定した数)
              = 実際にはりんごだったサンプルの中で、りんごだと推定して正解した割合
```
参考:  
[automl-natural-language-を使って文豪っぽさを推定する](https://www.apps-gcp.com/automl-natural-language-%E3%82%92%E4%BD%BF%E3%81%A3%E3%81%A6%E6%96%87%E8%B1%AA%E3%81%A3%E3%81%BD%E3%81%95%E3%82%92%E6%8E%A8%E5%AE%9A%E3%81%99%E3%82%8B/)

</br>

また、混同行列と特徴量の重要度を見ることもできる。  

混同行列:  
<img width=500 height=250 alt="06_confusion_matrix" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/06_confusion_matrix.png"/>
</br>
各行が正解ラベルで、それに対する各列が推定したラベルとなっています。  
今回の場合、「２」と推測したが実際には「１」だった割合が多くなってることがわかります。

特徴量の重要度:  
<img width=400 height=400 alt="06_feature_importance" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/06_feature_importance.png"/>
</br>
また、`duration`が最も重要なパラメータだったことがわかります。  
なお、この値はexportもできる。=> `out/feature_importance.csv`

### STEP 5: モデルのデプロイ&テスト
1, 「オンライン予測」から「モデルをデプロイ」する  
<img width=600 height=300 alt="07_deploy_model" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/07_deploy_model.png"/>
</br>

2, 「予測」を実行する  
<img width=600 height=300 alt="08_online_testing" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/08_online_testing.png"/>
</br>

予測の結果:  
<img width=300 height=200 alt="08_online_testing" src="https://github.com/ttogane/google_automl_table_demo/blob/main/screen_shots/08_online_testing.png"/>
</br>


## まとめ
Auto MLを使ってみると内容は単純でなんとなくMLに対しての理解が深まった。  
Google AutoMLはアップデートでどんどん種類も増えてるのでぜひ業務でも利用を模索していきたいと思う。