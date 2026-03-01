import sys
import random
import math
import ctypes
from src import cudaParticles
import time

## simulator object
particles = cudaParticles.CudaParticleMD()

## GPU device数
ndev = particles.nDevices()

## このスクリプトで使う単位は
## [1/Na g][Å][fs]
default_prams = {
    "cell": [0.0, 312.0, 0.0, 312.0, 0.0, 312.0],
    "kB": 8.3145e-7,
    "rho": 0.00748,
    "E": 1e-12,
    "rmax": 20.0,
    "thickness": 0.8,
    "mass": 44.01,
    "LJ_params": {
        "sigma": 3.941,
        "epsilon": 195.2,
    },
    # シミュレーション温度 [K]
    "Temp": 311.0,
    # タイムステップ [fs]
    "delta_t": 1.0,
    # 総シミュレーション時間 [fs]
    "total_time": 5000.0,
    # 出力間隔 [fs]
    "output_interval": 100.0,
    # 近接リスト更新間隔 [step]
    "neighbor_list_interval": 100,
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

def create_initial_state(params, log_params, dump_file):
    """
    calcMD_CO2.cu の createInitialState() 相当の処理
    """
    # パラメータ
    cell_list = params["cell"]
    Temp = params["Temp"]

    # kB in Å, fs, g/Na
    kB = params["kB"]
    rho = params["rho"]

    # セル長さ
    lx = cell_list[1] - cell_list[0]
    ly = cell_list[3] - cell_list[2]
    lz = cell_list[5] - cell_list[4]

    vol = lx * ly * lz
    N = int(vol * rho)

    # 格子数と格子間隔
    l0 = (vol / N) ** (1.0 / 3.0)
    n1 = int(lx / l0) + 1
    n2 = int(ly / l0) + 1
    n3 = int(lz / l0) + 1

    l1 = lx / n1
    l2 = ly / n2
    l3 = lz / n3

    # グローバルテーブルを構築
    G1 = cudaParticles.globalTable()

    carray3_t = ctypes.c_double * 3
    sigma_v = math.sqrt(kB * Temp / params["mass"])

    for i in range(N):
        pb = cudaParticles.ParticleBase()

        pb_r = carray3_t.from_address(int(pb.r))
        pb_v = carray3_t.from_address(int(pb.v))
        pb_a = carray3_t.from_address(int(pb.a))

        m1 = l1 * (i % n1 + 0.5)
        m2 = (((i - (i % n1)) // n1) % n2 + 0.5) * l2
        m3 = (int(i // (n1 * n2)) + 0.5) * l3

        pb_r[0] = m1
        pb_r[1] = m2
        pb_r[2] = m3

        # CO2
        pb.m = params["mass"]
        pb.type = 0
        pb_v[0] = random.gauss(0.0, sigma_v)
        pb_v[1] = random.gauss(0.0, sigma_v)
        pb_v[2] = random.gauss(0.0, sigma_v)
        pb_a[0] = pb_a[1] = pb_a[2] = 0.0
        pb.isFixed = False

        G1.push_back(pb)

    print(f"N= {G1.size()}", file=sys.stderr)

    # CUDA 側セル配列
    cell = cudaParticles.new_realArray(6)
    for i in range(6):
        cudaParticles.realArray_setitem(cell, i, cell_list[i])

    # GPU 側オブジェクト初期化
    ndev = particles.nDevices()
    particles.setup()

    for i in range(ndev):
        particles.setGPU(i)
        particles[i].setup(N)
        particles[i].setCell(cell)
        particles[i]._import(G1)
        particles[i].setM()

        particles[i].setE(params["E"])
        particles[i].kB = kB
        particles[i].timestep = 0
        particles[i].rmax2 = params["rmax"] * params["rmax"]
        particles[i].useNList()
        particles[i].thickness = params["thickness"]

    log_params({
        "N": G1.size(),
    })


def run(params, log_metrics, serialization_file, dump_file):
    """
    calcMD_CO2.cu の main() 相当のメインループ
    """

    # Lennard-Jones パラメータ設定 (CO2 1成分系)
    elemnum = 1
    LJpara = cudaParticles.VectorDouble(2 * elemnum * elemnum)
    LJpara[0] = params["LJ_params"]["sigma"]
    LJpara[1] = params["LJ_params"]["epsilon"] * params["kB"]

    ndev = particles.nDevices()
    for i in range(ndev):
        particles.setGPU(i)
        particles[i].setLJparams(LJpara, elemnum)

    # カットオフブロックと速度調整
    for i in range(ndev):
        particles.setGPU(i)
        particles[i].setupCutoffBlock(params["rmax"]+params["thickness"])
        particles[i].calcBlockID()
        particles[i].adjustVelocities(params["Temp"])

    # ブロックレンジ設定
    for i in range(ndev):
        particles[i].setBlockRange(particles[i].numBlocks(), ndev, i)

    # 初期状態のダンプ（必要なら）
    if dump_file:
        particles[0].openTMP(dump_file)
        particles[0].timestep = 0
        particles[0].getPosition()
        particles[0].putTMPtoO()
        particles[0].waitPutTMP()



    Temp = params["Temp"]
    delta_t = params["delta_t"]
    total_time = params["total_time"]
    output_interval_time = params["output_interval"]
    neighbor_list_interval = params["neighbor_list_interval"]

    stepnum = int(total_time / delta_t)
    ointerval = int(output_interval_time / delta_t)

    initstep = particles[0].timestep
    ndev = particles.nDevices()

    neighborListInterval = neighbor_list_interval


    start_time = time.time()
    print("start main loop", file=sys.stderr)

    for j in range(stepnum):
        # 力計算
        for i in range(ndev):
            particles.setGPU(i)
            if j % neighborListInterval == 0:
                particles[i].calcBlockID()
                particles[i].makeNList()
            particles[i].calcForce()

        # GPU 間で力を交換
        if ndev > 1:
            particles.exchangeForce()

        # 時間発展
        for i in range(ndev):
            particles.setGPU(i)
            t = particles[i].calcTemp()
            if i == 0:
                #print(f"T= {t}", file=sys.stderr)
                _temp = t

            if (t > Temp * 1.2) or math.isnan(t):
                print("abort", file=sys.stderr)
                if dump_file:
                    particles[0].waitPutTMP()
                    particles[0].closeTMP()
                sys.exit(1)

            if t > Temp * 1.1:
                scale = particles[i].scaleTemp(Temp)
                print(f"scaling temp by: {scale}", file=sys.stderr)

            particles[i].TimeEvolution(delta_t)
            particles[i].treatPeriodicCondition()

        # エネルギー計算とログ
        if (j + 1) % 25 == 0:
            K = particles[0].calcKineticE()
            P = particles[0].calcPotentialE()
            print(f"K-P\t{K}\t{P}", file=sys.stderr)
            if log_metrics is not None:
                log_metrics({
                    "kinetic_energy": K,
                    "potential_energy": P,
                    "temperature": _temp,
                    "elapsed_time": time.time() - start_time,
                    "time": (j + 1 + initstep) * delta_t,
                }, step=j + 1)

        # 位置のダンプ
        if (j + 1) % ointerval == 0 and dump_file:
            particles[0].timestep = j + 1 + initstep
            particles[0].getPosition()
            particles[0].putTMPtoO()

    if dump_file:
        particles[0].waitPutTMP()
        particles[0].closeTMP()

    particles.writeSerialization(serialization_file)
    print("Done.", file=sys.stderr)
