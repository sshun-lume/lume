# ルーメについて

ルーメはシミュレーション計算やシミュレーションプログラム開発の際に、計算試行の管理を行えるシミュレーションの援用環境です。
jupyterlabとmlflowを使って計算実行から結果の管理までを行います。


## 必要なもの

+ python
+ mlflow
+ jupyterlab
+ SQLiteあるいはMySQLその他SQL DB
+ AWS S3などのオブジェクトストレージ（mlflowのartifact配置にオブジェクトストレージを使う場合）

mlflowとjupyterlabはPythonのvirtualenv環境で`pip install mlflow`, `pip install jupyterlab`でインストールします。


### AWS EC2環境の場合

mlflow web UIとjupyterはssh tunnelingが必要です。
EC2のセキュリティ設定でmyIPからのインバウンドを登録しておいてローカルPCで以下のようにしてtunnel接続を張ります。

>ssh -i <秘密鍵> -N -f -L 8888:localhost:8888 ubuntu@<EC2パブリックIP>
ssh -i <秘密鍵> -N -f -L 5000:localhost:5000 ubuntu@<EC2パブリックIP>

AWS環境の時はmlflowのアーティファクトストレージをAWS S3に直接配置できます。

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

