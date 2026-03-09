from lammps import lammps
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
    lmp.command(f"units {params['units']}")
    lmp.command(f"atom_style {params['atom_style']}")
    lmp.command(f"read_data {params['data_dir']}/{params['data_file']}")
    
    lmp.file(f"{params['data_dir']}/{params['input_file']}")
    lmp.command(f"dump 1 all custom 100 {dump_file}.* id type xs ys zs ix iy iz vx vy vz fx fy fz")


def run(params, log_metrics, restart_file, dump_file):
    iter = int(params['run_steps'] / params['thermo_step'])
    for i in range(iter):
        lmp.command(f"run {params['thermo_step']}")
        pe = lmp.get_thermo("pe")
        ke = lmp.get_thermo("ke")
        temp = lmp.get_thermo("temp")
        press = lmp.get_thermo("press")
        etotal = lmp.get_thermo("etotal")
        log_metrics({
            "potential_energy": pe,
            "kinetic_energy": ke,
            "temperature": temp,
            "pressure": press,
            "total_energy": etotal,
        }, step=(i+1)*params['thermo_step'])

    lmp.command(f"write_restart {restart_file}")