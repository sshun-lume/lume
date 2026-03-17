from lammps import lammps
import math

default_prams = {
    "data_file": "data.lmp",
    "input_file": "in.lmp",
    "log_file": "lammps",
    "data_dir": "data",
    "thermo_step": 100,
    "run_steps": 100,
}

def log_GPU_info(log_params):
    pass

def load_previous_state(params, prev_state):
    global lmp
    lmp = lammps()
    lmp.command(f"log log.{params['log_file']}")

    lmp.command(f"variable RestartFrom world {prev_state}")
    
    lmp.command(f"variable ThermoStep world {params['thermo_step']}")

    lmp.file(f"{params['data_dir']}/{params['input_file']}")

def create_initial_state(params, log_params, snapshot_file):
    global lmp
    lmp = lammps()
    lmp.command(f"log log.{params['log_file']}")

    lmp.command(f"variable DataFile world {params['data_dir']}/{params['data_file']}")
    lmp.command(f"variable ThermoStep world {params['thermo_step']}")
    lmp.command(f"variable Snapshot world {snapshot_file}")

    lmp.file(f"{params['data_dir']}/{params['input_file']}")


def run(params, log_metrics, restart_file, dump_file):
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    iter = math.ceil(params['run_steps'] / params['thermo_step'])
    for i in range(iter):
        lmp.command(f"run {params['thermo_step']}")
        thermo = lmp.last_thermo()
        if rank == 0:
            log_metrics(thermo, step=(i+1)*params['thermo_step'])

    lmp.command(f"write_restart {restart_file}")

def store_artifacts(recipe, log_artifact, archive_command, cleanup=False):
    import os, glob
    if 'snapshots' in recipe:
        for snapshot in recipe['snapshots']:
            for snapshot_file in glob.glob(f"{snapshot}.*"):
                if snapshot_file.endswith('.xz'):
                    continue
                os.system(f"{archive_command} {snapshot_file}")
                log_artifact(f"{snapshot_file}.xz", artifact_path='snapshots')
            if cleanup:
                os.system(f"rm -f {snapshot}.*")
    
    if 'log' in recipe:
        log_artifact(f'{recipe["log"]}', artifact_path='log')
        if cleanup:
            os.system(f"rm -f {recipe["log"]}")
        
    if 'restarts' in recipe:
        for restart in recipe['restarts']:
            log_artifact(restart, artifact_path='restarts')
            if cleanup:
                os.system(f"rm -f {restart}")

    if 'dumpfiles' in recipe:
        for dumpfile in recipe['dumpfiles']:
            try:
                os.system(f"{archive_command} {dumpfile}")
                log_artifact(f"{dumpfile}.xz", artifact_path='dumpfiles')
                if cleanup:
                    os.system(f"rm -f {dumpfile}.xz")
            except:
                pass
