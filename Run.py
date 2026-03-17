import importlib
import mlflow
import os, sys, json
from utils.info_utils import MLflowEnvLogger
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


RUN_SCRIPT = os.getenv("RUN_SCRIPT")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
RUN_NAME = os.getenv("RUN_NAME")
ARCHIVE_COMMAND = os.getenv("ARCHIVE_COMMAND")
PREV_RUNID = os.getenv("PREV_RUNID")
RESTART_FROM = os.getenv("RESTART_FROM")
mlflow_run_id = sys.argv[1]

sim_params = json.loads(os.getenv("SIM_PARAMS"))
dump_files = json.loads(os.getenv("DUMP_FILES"))
snapshots = os.getenv("SNAPSHOTS")
restart_files = json.loads(os.getenv("RESTART_FILES"))

restart = f'restart.{RUN_NAME}'
restart_files.append(restart)

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
    if rank == 0:
        sim.log_GPU_info(mlflow.set_tags)
        mlflow.log_params(sim_params)
        mlflow.log_params({
            "dump_files": dump_files,
            "snapshots": snapshots,
            "restart_files": restart_files,
        })
        mlflow.set_tags(MLflowEnvLogger.log_all_env_tags())

        mlflow.log_params({
            "dump_files": dump_files,
            "snapshots": snapshots,
            "restart_files": restart_files,
        })

    if PREV_RUNID:
        prev_state_path = mlflow.artifacts.download_artifacts(
            run_id=PREV_RUNID,
            artifact_path="restarts",    # 取得したいアーティファクト内のパス
        )
        files = os.listdir(prev_state_path)
        if RESTART_FROM != '':
            RESTART_FILE = RESTART_FROM
        elif len(files) > 0:
            RESTART_FILE = files[0]
        else:
            sys.exit(0)
        sim.load_previous_state(sim_params, f'{prev_state_path}/{RESTART_FILE}')
        os.system(f"rm -rf {prev_state_path}")
        if rank == 0:
            mlflow.set_tag("PREV_RUNID", PREV_RUNID)
    else:
        sim.create_initial_state(sim_params, mlflow.log_params, dump_files[1])

    ##
    ## シミュレーション処理　メインループ
    ##
    sim.run(sim_params, mlflow.log_metrics, restart, dump_files)

finally:
    ##
    ## シミュレーション処理　結果の登録・保存
    ##
    recipe = {
        "dumpfiles": dump_files,
        "snapshots": [snapshots],
        "restarts": restart_files,
    }
    if "log_file" in sim_params:
        recipe.update({"log": f'log.{sim_params["log_file"]}'})

    if rank == 0:
        sim.store_artifacts(recipe, mlflow.log_artifact, f"{ARCHIVE_COMMAND} -t", cleanup=True)

if rank == 0:
    mlflow.log_artifact(f'{RUN_NAME}.out', artifact_path='output')
    mlflow.log_artifact(f'{RUN_NAME}.err', artifact_path='output')

    mlflow.end_run()
