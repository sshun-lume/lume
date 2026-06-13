import mlflow
import os, sys, json
import importlib


mlflow_run_id = sys.argv[1]

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.log_artifact(f'{os.getenv("RUN_NAME")}.out.txt', artifact_path='output', run_id=mlflow_run_id)
mlflow.log_artifact(f'{os.getenv("RUN_NAME")}.err', artifact_path='output', run_id=mlflow_run_id)

artifacts = {
    "dumpfiles": json.loads(os.getenv("DUMP_FILES")),
    "snapshots": [os.getenv("SNAPSHOTS")],
    "restarts": json.loads(os.getenv("RESTART_FILES")),
}
if "log_file" in json.loads(os.getenv("SIM_PARAMS")):
    artifacts.update({"log": f'log.{json.loads(os.getenv("SIM_PARAMS"))["log_file"]}'})


#sim = importlib.import_module(f'scripts.{os.getenv("RUN_SCRIPT")}')
#sim.store_artifacts(artifacts, mlflow.log_artifact, f"{os.getenv("ARCHIVE_COMMAND")} -t", cleanup=os.getenv("ARTIFACTS_CLEANUP") == "True")


mlflow.end_run(status='FAILED', run_id=mlflow_run_id)