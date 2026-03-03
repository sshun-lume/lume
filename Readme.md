# ルーメについて

ルーメはシミュレーション計算やシミュレーションプログラム開発の際に、計算試行の管理を行えるシミュレーションの援用環境です。
jupyterlabとmlflowを使って計算実行から結果の管理までを行います。


## 必要なもの

+ python
+ mlflow
+ jupyterlab
+ SQLiteあるいはMySQLその他SQL DB
+ AWS S3などのオブジェクトストレージ（mlflowのartifact配置にオブジェクトストレージを使う場合）
+ slurm（ジョブキューイングを行う場合）

mlflowとjupyterlabはPythonのvirtualenv環境で`pip install mlflow`, `pip install jupyterlab`でインストールします。
slurmは`apt`コマンドでインストールします。

>sudo apt install slurm-wlm

### AWS EC2環境の場合

mlflow web UIとjupyterはssh tunnelingが必要です。
EC2のセキュリティ設定でmyIPからのインバウンドを登録しておいてローカルPCで以下のようにしてtunnel接続を張ります。

>ssh -i <秘密鍵> -N -f -L 8888:localhost:8888 ubuntu@<EC2パブリックIP>  
ssh -i <秘密鍵> -N -f -L 5000:localhost:5000 ubuntu@<EC2パブリックIP>

AWS環境の時はmlflowのアーティファクトストレージをAWS S3に直接配置できます。

slurmを使用する場合でhostnameを固定したい場合は以下で変更します。

>sudo hostnamectl set-hostname <新しいホスト名>


## 構成

### 1台構成の場合

結果の管理を行うmlflowと計算の実行を開始するjupyterlab、そしてシミュレーション計算それ自体を一つのホストで行う構成です。
mlflowのtracking serverはローカルファイルシステムを指定し、確認用のweb UIだけを起動します。

それぞれ
>jupyter lab --ip=0.0.0.0 --port=8888 --no-browser  
mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////path/to/mlruns/mlflow.db

で起動します。

`Run.ipynb`がシミュレーションの実行を行うjupyterlab notebookです。
シミュレーション本体は`scripts/`に配置したpython scriptでシミュレーションパラメータ、初期化、メインループを記述しておき、
jupyterlab notebookのkernelで実行しています。

### アーティファクトストレージをS3に配置する場合

mlflowのアーティファクトストレージにS3 bucketを指定することができます。
mlflowのexperimentごとにアーティファクトストレージを設定できるので`MLFLOW_STORAGE`にS3 bucketを指定して新しい`MLFLOW_EXP_TYPE`で実験を登録するとアーティファクトはS3に保存されます。

この場合はEC2からS3への書き込み権限が必要です

#### S3への書き込み権限を計算を行うEC2インスタンスに与える手順

1. 「IAM > ポリシー」で`ポリシーの作成`を行う
  1.1. `サービスを選択`でS3を選択する
  1.2. `アクション許可`で、`書き込み`では`PutObject`を、`読み取り`で`GetObject`を、`リスト`では`ListBucket`をチェックする
  1.3. `リソース`でbucketのARNを追加して、その特定リソースのみ許可する
  `arn:aws:s3:::バケット名`および`arn:aws:s3:::バケット名/*`の2つを登録する。
  1.4. 任意のポリシー名と説明をつけて保存
2. 「IAM > ロール」で`ロールを作成`を行う
  2.1. エンティティタイプ: AWSのサービス、ユースケース: EC2
  2.2. 許可ポリシーで上で作成したポリシーを選択
  2.3. ロール名と説明をつけて保存
3. EC2コンソールで当該インスタンスを選択してアクション>セキュリティ>IAMロールを変更
　　3.1. 上で作成したロールをアタッチする

### 2台以上で構成する場合（mlflowをserver modeで起動する場合）

mlflowを以下でserver modeで起動しておきます。
>mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:////path/to/mlserver/mlflow.db

ノートブック中のmlflowのtracking URIを以下のように指定します。
>MLFLOW_TRACKING_URI = "http://<サーバのIP>:5000"

### slurmなどを使って計算ジョブをキューイングする場合

`Batch.ipynb`のノートブックからジョブを投入します。
最後のセルに`mlflow.end_run()`がありますが、正常にジョブ投入できている時は実行しないように注意。
