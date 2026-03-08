from lammps import lammps
lmp = lammps()

default_prams = {
    "data_file": "data.lmp",
    "input_file": "in.lmp",
    "log_file": "lammps",
    "data_dir": "data",
    "run_steps": 100,
}

def log_GPU_info(log_params):
    pass

def load_previous_state(prev_state):
    pass

def create_initial_state(params, log_params, dump_file):
    lmp.command(f"log log.{params['log_file']}")
    lmp.command("units lj")
    lmp.command("atom_style molecular")
    lmp.command(f"read_data {params['data_dir']}/{params['data_file']}")
    
    lmp.file(f"{params['data_dir']}/{params['input_file']}")
    lmp.command(f"dump 1 all custom 100 {dump_file}.* id type xs ys zs ix iy iz vx vy vz fx fy fz")


def run(params, log_metrics, serialization_file, dump_file):
    lmp.command(f"run {params['run_steps']}")
    lmp.command("write_restart " + serialization_file)

