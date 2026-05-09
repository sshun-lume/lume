from lammps import lammps
import math

default_prams = {
    "data_file": "data.lmp",
    "input_file": "in.lmp",
    "log_file": "lammps",
    "data_dir": "data",
    "delta_t": 1.0,
    "thermo_step": 100,
    "dump_step": 100,
    "run_steps": 100,
    "lmp_suffix": "opt",    # "gpu", "omp", "opt", "off"
    "pk_num": 1,
}

def log_GPU_info(log_params):
    pass

def load_previous_state(params, prev_state):
    global lmp
    lmp = lammps()
    lmp.command(f"log log.{params['log_file']}")
    if params["lmp_suffix"] != "off":
        lmp.command(f"suffix {params["lmp_suffix"]}")
        if params["lmp_suffix"] != "opt":
            lmp.command(f"package {params["lmp_suffix"]} {params["pk_num"]}")

    lmp.command(f"variable        DeltaT world {params['delta_t']}")
    lmp.command(f"variable RestartFrom world {prev_state}")
    
    lmp.command(f"variable ThermoStep world {params['thermo_step']}")
    lmp.command(f"variable DumpStep world {params['dump_step']}")

    lmp.file(f"{params['data_dir']}/{params['input_file']}")

def create_initial_state(params, log_params, snapshot_file):
    global lmp
    lmp = lammps()
    lmp.command(f"log log.{params['log_file']}")
    if params["lmp_suffix"] != "off":
        lmp.command(f"suffix {params["lmp_suffix"]}")
        if params["lmp_suffix"] != "opt":
            lmp.command(f"package {params["lmp_suffix"]} {params["pk_num"]}")

    lmp.command(f"variable        DeltaT world {params['delta_t']}")
    lmp.command(f"variable DataFile world {params['data_dir']}/{params['data_file']}")

    lmp.command(f"variable ThermoStep world {params['thermo_step']}")
    lmp.command(f"variable DumpStep world {params['dump_step']}")
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

def store_artifacts(artifacts, log_artifact, archive_command, cleanup=False):
    import os, glob
    if 'snapshots' in artifacts:
        for snapshot in artifacts['snapshots']:
            if snapshot != "" and snapshot != None:
                for snapshot_file in glob.glob(f"{snapshot}.*"):
                    if snapshot_file.endswith('.xz'):
                        continue
                    os.system(f"{archive_command} {snapshot_file}")
                    log_artifact(f"{snapshot_file}.xz", artifact_path='snapshots')
                if cleanup:
                    os.system(f"rm -f {snapshot}.*")

    if 'log' in artifacts:
        log_artifact(f'{artifacts["log"]}', artifact_path='log')
        if cleanup:
            os.system(f"rm -f {artifacts["log"]}")
        
    if 'restarts' in artifacts:
        for restart in artifacts['restarts']:
            log_artifact(restart, artifact_path='restarts')
            if cleanup:
                os.system(f"rm -f {restart}")

    if 'dumpfiles' in artifacts:
        for dumpfile in artifacts['dumpfiles']:
            try:
                os.system(f"{archive_command} {dumpfile}")
                log_artifact(f"{dumpfile}.xz", artifact_path='dumpfiles')
                if cleanup:
                    os.system(f"rm -f {dumpfile}.xz")
            except:
                pass
