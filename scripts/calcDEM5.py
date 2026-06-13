import sys
import random
import math
import ctypes
from src import cudaParticles
import time

## simulator object
particles = cudaParticles.CudaParticleDEM()

## GPU device数
ndev = particles.nDevices()


## 下で使う計算変数
e = 0.85
_gamma = - math.log(e) / math.sqrt(math.pi*math.pi + math.log(e)*math.log(e)) * 2;

## units used in this simulation are
## [g][cm][s]
default_prams = {
    "R0": 0.50,
    "cell": [0.0, 60.0, 0.0, 60.0, 0.0, 50.0],
    "density": 7.874,
    "DEM_params": {
        "E": 2.11e10,
        "mu": 0.40,
        "sigma": 0.29,
        "gamma": _gamma,
        "mu_r": 0.10,
    },
    "cutoff_block_factor": 0.9,
    "total_time": 1.50,
    "intaval": 0.005,
    "initDeltaT": 0.000008,
    "ulim": 0.01 * 4,
    "param_g": 9.8e2,
}

def log_GPU_info(log_params):
    for i in range(ndev):
        particles.setGPU(i)
        log_params({
            f"GPU_{i}_name": particles.getName(),
            f"GPU_{i}_capability": particles.getCapability(),
            f"GPU_{i}_numMP": particles.getNumMP(),
            f"GPU_{i}_numCpMP": particles.getNumCpMP(),
            f"GPU_{i}_globalMemSize": f'{particles.getMemSize() /1024 /1024 /1024} [GB]',
            f"GPU_{i}_numTh": particles.getNumTh(),
            f"GPU_{i}_maxGrid": particles.getMaxGrid(),
            f"GPU_{i}_shMem": f'{particles.getShMem() /1024} [KB]',
        })

def load_previous_state(_, prev_state):
    particles.setup()
    particles.readSerialization(prev_state)


def create_initial_state(params, log_params, dump_file):
    G1 = cudaParticles.globalTable()

    cell = cudaParticles.new_realArray(6)
    for i in range(6):
        cudaParticles.realArray_setitem(cell, i, params["cell"][i])

    R0 = params["R0"]
    lunit = 2 * params["R0"]
    WeighFe = params["density"] * 4.0 / 3.0 * math.pi * R0*R0*R0

    # border particles
    L1 = params["cell"][1]
    L2 = params["cell"][3]
    L3 = 10.0
    lsize1 = int(L1 / lunit + 1)
    lsize2 = int(L2 / lunit + 1)
    lsize3 = int(L3 / lunit + 1)

    carray3_t = ctypes.c_double * 3

    for k in range(lsize3):
        for i in range(lsize1):
            for j in range(lsize2):

                if ( ( (i<1) or (lsize1-2<i) or (j<1) or (lsize2-2<j) )
                     or
                     (k==0)
                ):
                    pb = cudaParticles.ParticleBase()

                    pb_r = carray3_t.from_address(int(pb.r))
                    pb_v = carray3_t.from_address(int(pb.v))
                    pb_a = carray3_t.from_address(int(pb.a))
                    if ( (0<i) and (i<lsize1-1) and (0<j) and (j<lsize2-1) ):
                        pb_r[0] = i*lunit + random.gauss(0.0, R0/100.0)
                        pb_r[1] = j*lunit + random.gauss(0.0, R0/100.0)
                    else:
                        pb_r[0] = i*lunit
                        pb_r[1] = j*lunit

                    pb_r[2] = k*lunit
                    pb.m = WeighFe
                    pb_v[0] = pb_v[1] = pb_v[2] = 0.0
                    pb_a[0] = pb_a[1] = pb_a[2] = 0.0
                    pb.isFixed = True
                    pb.type = 0
                    G1.push_back(pb)

    N1=G1.size()
    print("N=", N1, file=sys.stderr)

    # moving
    L4 = params["cell"][5]
    L5 = 20.0
    P1 = int(L5 / lunit / 1.3)
    P2 = int(L4 / lunit - 2)

    for k in range(P2):
        for i in range(P1):
            for j in range(P1):
                pb = cudaParticles.ParticleBase()
                pb_r = carray3_t.from_address(int(pb.r))
                pb_v = carray3_t.from_address(int(pb.v))
                pb_a = carray3_t.from_address(int(pb.a))
                pb_r[0] = (i*1.3)*lunit + random.gauss(0.0, R0/10.0) + L5
                pb_r[1] = (j*1.3)*lunit + random.gauss(0.0, R0/10.0) + L5
                pb_r[2] = (k+1  )*lunit + lunit/3
                pb.m = WeighFe
                pb_v[0] = pb_v[1] = pb_v[2] = 0.0
                pb_a[0] = pb_a[1] = pb_a[2] = 0.0
                pb.isFixed = False
                pb.type = 1
                G1.push_back(pb)

    N = G1.size()
    print("N=", N, file=sys.stderr)


    ndev = particles.nDevices()
    particles.setup()

    DP = params["DEM_params"]
    for i in range(ndev):
        particles.setGPU(i)

        particles[i].setup(N)
        particles[i].setCell(cell)

        particles[i]._import(G1)

        particles[i].setDEMProperties(DP['E'], DP['mu'], DP['sigma'], DP['gamma'], DP['mu_r'], R0)
        particles[i].setInertia(R0)
        particles[i].setupCutoffBlock(R0*2/math.sqrt(3.0)*params["cutoff_block_factor"], False)

        # putTMPselected
        particles[i].setupSelectedTMP(N1, N-N1, 0, N1)


        particles[i].checkPidRange(i, ndev, N1, N)

    log_params({
        "N": G1.size(),
        "border_particles": N1,
        "moving_particles": N - N1,
    })

    particles[0].timestep = 0
    particles[0].putUnSelected(dump_file)

    for i in range(ndev):
        particles.setGPU(i)
        particles[i].calcBlockID()


def run(params, log_metrics, serialization_file, dump_files):

    particles[0].getSelectedTypeID()
    particles[0].getSelectedPosition()
    particles[0].openTMP(dump_files[0])
    particles[0].putTMPtoO()
    particles[0].waitPutTMP()


    total_time  = params["total_time"]
    intaval  = params["intaval"]
    initstep = particles[0].timestep
    initDeltaT = params["initDeltaT"]
    ulim = params["ulim"]
    llim = ulim / 16.0
    res = cudaParticles.VectorInt(ndev)

    R0 = params["R0"]
    param_g = params["param_g"]

    thistime = cudaParticles.AdaptiveTimeDEM()
    nextoutput = intaval
    thistime.init(initDeltaT)
    thistime.statOutput = True

    thistime.PrintStat(0)
    print("End Time: ", total_time, file=sys.stderr)

    for i in range(ndev):
        particles.setGPU(i)
        particles[i].selectBlocks()
        particles[i].calcVinit(initDeltaT)

    j = 0
    start_time = time.time()

    while thistime() < total_time:
        if thistime.isRollbacking():
            print("now rollbacking:", end="", file=sys.stderr)

        if j % 50 == 0:
            print(j, end=" ", file=sys.stderr)
            log_metrics({
                "elapsed_time": time.time() - start_time,
                "time": thistime(),
                "delta_t": thistime.currentDeltaT(),
            }, step=j)

        for i in range(ndev):
            res[i] = 0
            particles.setGPU(i)
            particles[i].calcBlockID()

            particles[i].selectBlocks()
            particles[i].calcForce(thistime.currentDeltaT())

        if ndev > 1:
            particles.exchangeForceSelected(0)
            particles.exchangeForceSelected(2)

        for i in range(ndev):
            particles.setGPU(i)
            particles[i].calcAcceleration()
            particles[i].addAccelerationZ(-param_g)
            particles[i].TimeEvolution(thistime.currentDeltaT())
            res[i] = particles[i].inspectVelocity((2*R0)/thistime.currentDeltaT(), ulim, llim)

        j = thistime.Progress(particles, res, j)


        for i in range(ndev):
            particles.setGPU(i)
            particles[i].treatRefrectCondition()

        if thistime() >= nextoutput:
            print("({})".format(thistime()), file=sys.stderr)
            nextoutput = nextoutput + intaval
            particles[0].timestep = j+1+initstep
            particles[0].getSelectedPosition()
            particles[0].putTMPtoO()


    particles[0].waitPutTMP()
    thistime.PrintStat(j)
    log_metrics({
        "elapsed_time": time.time() - start_time,
        "time": thistime(),
        "delta_t": thistime.currentDeltaT(),
    }, step=j)

    particles.writeSerialization(serialization_file)
    print("Done.")

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
            if restart != "" and restart != None and os.path.exists(restart):
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
