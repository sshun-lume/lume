import importlib
import mlflow
import os, sys, json


RUN_SCRIPT = os.getenv("RUN_SCRIPT")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RUN_NAME = os.getenv("RUN_NAME")
ARCHIVE_COMMAND = os.getenv("ARCHIVE_COMMAND")
PREV_RUNID = os.getenv("PREV_RUNID")
mlflow_run_id = sys.argv[1]

sim_params = json.loads(os.getenv("SIM_PARAMS"))
dump_files = json.loads(os.getenv("DUMP_FILES"))

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
    sim.log_GPU_info(mlflow.log_params)
    mlflow.log_params(sim_params)
    
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
        sim.create_initial_state(sim_params, mlflow.log_params, dump_files[1])

    ##
    ## シミュレーション処理　メインループ
    ##
    archive = f'_{RUN_NAME}'
    sim.run(sim_params, mlflow.log_metrics, archive, dump_files[0])

    mlflow.log_artifact(archive, artifact_path='final_state')

finally:
    ##
    ## シミュレーション処理　結果の登録・保存
    ##
    for dump_file in dump_files:
        if dump_file:
            os.system(f"{ARCHIVE_COMMAND} {dump_file}")
            mlflow.log_artifact(f"{dump_file}.xz", artifact_path='dump')

mlflow.log_artifact(f'{RUN_NAME}.out', artifact_path='output')
mlflow.log_artifact(f'{RUN_NAME}.err', artifact_path='output')

mlflow.end_run()
