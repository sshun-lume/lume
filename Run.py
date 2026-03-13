import importlib
import mlflow
import os, sys, json
from utils.info_utils import MLflowEnvLogger


RUN_SCRIPT = os.getenv("RUN_SCRIPT")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RUN_NAME = os.getenv("RUN_NAME")
ARCHIVE_COMMAND = os.getenv("ARCHIVE_COMMAND")
PREV_RUNID = os.getenv("PREV_RUNID")
mlflow_run_id = sys.argv[1]

sim_params = json.loads(os.getenv("SIM_PARAMS"))
dump_files = json.loads(os.getenv("DUMP_FILES"))
snapshots = os.getenv("SNAPSHOTS")

##
##
##
sim = importlib.import_module(f'scripts.{RUN_SCRIPT}')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


mlflow_run = mlflow.start_run(
    run_id=mlflow_run_id,
    log_system_metrics=True)

try:
    ##
    ## シミュレーション処理　シミュレーション初期化
    ##
    sim.log_GPU_info(mlflow.set_tags)
    mlflow.log_params(sim_params)
    mlflow.set_tags(MLflowEnvLogger.log_all_env_tags())
    
    if PREV_RUNID:
        prev_state_path = mlflow.artifacts.download_artifacts(
            run_id=PREV_RUNID,
            artifact_path="final_state",    # 取得したいアーティファクト内のパス
        )
        files = os.listdir(prev_state_path)
        sim.load_previous_state(f'{prev_state_path}/{files[0]}')
        os.system(f"rm -rf {prev_state_path}")
        mlflow.log_param("prev_run_id", PREV_RUNID)
    else:
        sim.create_initial_state(sim_params, mlflow.log_params, snapshots)

    ##
    ## シミュレーション処理　メインループ
    ##
    restart = f'restart.{RUN_NAME}'
    sim.run(sim_params, mlflow.log_metrics, restart, dump_files)

finally:
    ##
    ## シミュレーション処理　結果の登録・保存
    ##
    recipe = {
        "dumpfiles": dump_files,
        "snapshots": [snapshots],
        "log": f'log.{sim_params["log_file"]}',
        "restarts": [restart],
    }

    sim.store_artifacts(recipe, mlflow.log_artifact, f"{ARCHIVE_COMMAND} -t", cleanup=True)

mlflow.log_artifact(f'{RUN_NAME}.out', artifact_path='output')
mlflow.log_artifact(f'{RUN_NAME}.err', artifact_path='output')

mlflow.end_run()
