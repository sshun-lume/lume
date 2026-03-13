from lammps import lammps
import math
import os, glob
lmp = lammps()

default_prams = {
    "units": "lj",
    "atom_style": "molecular",
    "data_file": "data.lmp",
    "input_file": "in.lmp",
    "log_file": "lammps",
    "data_dir": "data",
    "thermo_step": 100,
    "run_steps": 100,
}

def log_GPU_info(log_params):
    pass

def load_previous_state(prev_state):
    pass

def create_initial_state(params, log_params, dump_file):
    lmp.command(f"log log.{params['log_file']}")

    lmp.command(f"variable DataFile world {params['data_dir']}/{params['data_file']}")
    lmp.command(f"variable ThermoStep world {params['thermo_step']}")
    lmp.command(f"variable DumpFile world {dump_file}")

    lmp.file(f"{params['data_dir']}/{params['input_file']}")


def run(params, log_metrics, restart_file, dump_file):
    iter = math.ceil(params['run_steps'] / params['thermo_step'])
    for i in range(iter):
        lmp.command(f"run {params['thermo_step']}")
        thermo = lmp.last_thermo()
        log_metrics(thermo, step=(i+1)*params['thermo_step'])

    lmp.command(f"write_restart {restart_file}")

def store_artifacts(recipe, log_artifact, archive_command, cleanup=False):
    if recipe['snapshots']:
        for snapshot in recipe['snapshots']:
            for snapshot_file in glob.glob(f"{snapshot}.*"):
                if snapshot_file.endswith('.xz'):
                    continue
                os.system(f"{archive_command} {snapshot_file}")
                log_artifact(f"{snapshot_file}.xz", artifact_path='snapshots')
            if cleanup:
                os.system(f"rm -f {snapshot}.*")
    
    if recipe['log']:
        log_artifact(f'{recipe["log"]}', artifact_path='log')
        if cleanup:
            os.system(f"rm -f {recipe["log"]}")
        
    if recipe['restarts']:
        for restart in recipe['restarts']:
            log_artifact(restart, artifact_path='restarts')
            if cleanup:
                os.system(f"rm -f {restart}")
