import sys
import random
import math
import ctypes
from src import cudaParticles
import time

## simulator object
particles = cudaParticles.CudaParticleSPH_NS()

## GPU device数
ndev = particles.nDevices()

## units used in this simulation are
## [g][cm][s]
default_prams = {
    "cell": [0.0, 20.0, 0.0, 20.0, 0.0, 25.0],
    # C++版 testSPH6.cu に対応する物理パラメータ
    "lunit": 0.4,
    "sph_h_factor": 0.5938629,
    "rho0_coef": 20.0,          # rho_0 = rho0_coef / (4/3*pi*h^3)
    "mu_fluid": 8.9e-3,
    "mu_solid": 1.0e5,
    "c_fluid": 1.5e3 / 2.0,     # 1500m/s = 1.5e5 cm/s (water)
    "c_solid": 5.44e3 / 2.0,    # 5440m/s (glass)
    "deltaT": 0.000050,
    "stepmax": 1.50,            # [s]
    "intaval": 0.005,           # [s]
    "param_g": 9.8e2,           # [cm/s^2]
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
    G1 = cudaParticles.globalTable()

    cell = cudaParticles.new_realArray(6)
    for i in range(6):
        cudaParticles.realArray_setitem(cell, i, params["cell"][i])

    # C++ testSPH6.cu のパラメータをPythonに移植
    lunit = params["lunit"]
    sph_h = lunit / params["sph_h_factor"]

    # 密度・質量
    rho0_coef = params["rho0_coef"]
    rho_0 = rho0_coef / (4.0 / 3.0 * math.pi * sph_h * sph_h * sph_h)
    m_0 = 1.0 / rho_0

    print("SPH kernel h =", sph_h,
          "lunit =", lunit,
          "mean number density =", rho_0,
          file=sys.stderr)
          
    log_params({
        "SPH_kernel_h": sph_h,
        "mean_number_density": rho_0,
        "m_0": m_0
    })

    # グリッドサイズ（C++版の L1, L2, L3, H5, H10, H15 と対応）
    L1 = int(20.0 / lunit - 1)
    L2 = int(20.0 / lunit - 1)
    L3 = int(25.0 / lunit - 1)
    lunit_h = lunit / 2.0

    H5 = int(5.0 / lunit - 1)
    H10 = int(10.0 / lunit - 1)
    H15 = int(15.0 / lunit - 1)

    carray3_t = ctypes.c_double * 3

    # 境界粒子: 20x20x25 cm box
    for iz in range(L3):
        for iy in range(L2):
            for ix in range(L1):
                # i: Z(iz), j: Y(iy), k: X(ix)
                if (
                    ((iz == 0) and (iy < L2 / 2) and (ix < L1 / 2)) or
                    ((iz == H5) and (iy < L2 / 2) and (ix >= L1 / 2)) or
                    ((iz == H10) and (iy >= L2 / 2) and (ix >= L1 / 2)) or
                    ((iz == H15) and (iy >= L2 / 2) and (ix < L1 / 2)) or  # floor
                    ((ix == 0) and (iy < L2 / 2)) or
                    ((ix == 0) and (iy >= L2 / 2) and (iz >= H15)) or
                    ((ix == L1 - 1) and (iy < L2 / 2) and (iz >= H5)) or
                    ((ix == L1 - 1) and (iy >= L2 / 2) and (iz >= H10)) or
                    ((iy == 0) and (ix < L1 / 2)) or
                    ((iy == 0) and (ix >= L1 / 2) and (iz >= H5)) or
                    ((iy == L2 - 1) and (ix < L1 / 2) and (iz >= H15)) or
                    ((iy == L2 - 1) and (ix >= L1 / 2) and (iz >= H10)) or  # wall
                    ((iy == L2 / 2) and (ix < L1 / 2)) or
                    ((iy == L2 / 2) and (ix >= L1 / 2) and (iz >= H5) and (iz < H10)) or
                    ((ix == L1 / 2) and (iy < L2 / 2) and (iz < H5)) or
                    ((ix == L1 / 2) and (iy >= L2 / 2) and (iz >= H10) and (iz < H15))
                ):
                    # 同じ位置に2つの境界粒子を配置
                    for _ in range(2):
                        pb = cudaParticles.ParticleBase()
                        pb_r = carray3_t.from_address(int(pb.r))
                        pb_v = carray3_t.from_address(int(pb.v))
                        pb_a = carray3_t.from_address(int(pb.a))

                        pb_r[0] = ix * lunit + lunit_h
                        pb_r[1] = iy * lunit + lunit_h
                        pb_r[2] = iz * lunit + lunit_h

                        pb.m = m_0 * 2.5  # x2.5 by water
                        pb_v[0] = pb_v[1] = pb_v[2] = 0.0
                        pb_a[0] = pb_a[1] = pb_a[2] = 0.0
                        pb.isFixed = True
                        pb.type = 0
                        G1.push_back(pb)

    N1 = G1.size()
    print("N=", N1, file=sys.stderr)

    # 流体粒子
    # 壁位置 Y:L2/2, L2; X:0, L1/2
    for iz in range(H15 + 2, L3):
        for iy in range(int(L2 / 2) + 2, L2 - 1):
            for ix in range(2, int(L1 / 2) - 1):
                pb = cudaParticles.ParticleBase()
                pb_r = carray3_t.from_address(int(pb.r))
                pb_v = carray3_t.from_address(int(pb.v))
                pb_a = carray3_t.from_address(int(pb.a))

                # C++版では正規分布。ここでは random.gauss を用いる
                sigma = lunit_h / 50.0
                pb_r[0] = ix * lunit + random.gauss(0.0, sigma)
                pb_r[1] = iy * lunit + random.gauss(0.0, sigma)
                pb_r[2] = iz * lunit + random.gauss(0.0, sigma)

                pb.m = m_0
                pb_v[0] = pb_v[1] = pb_v[2] = 0.0
                pb_a[0] = pb_a[1] = pb_a[2] = 0.0
                pb.isFixed = False
                pb.type = 1
                G1.push_back(pb)

    N = G1.size()
    print("N=", N, file=sys.stderr)

    log_params({
        "N": G1.size(),
        "border_particles": N1,
        "moving_particles": N - N1,
    })

    # 物性パラメータ mu, c1 を粒子ごとに設定
    mu_fluid = params["mu_fluid"]
    mu_solid = params["mu_solid"]
    c_fluid = params["c_fluid"]
    c_solid = params["c_solid"]

    mu = [mu_fluid] * N
    c1 = [c_fluid] * N
    for i in range(N1):
        mu[i] = mu_solid
        c1[i] = c_solid

    mu_vec = cudaParticles.VectorDouble(mu)
    c1_vec = cudaParticles.VectorDouble(c1)

    ndev = particles.nDevices()
    particles.setup()

    for i in range(ndev):
        particles.setGPU(i)
        particles[i].setup(N)
        particles[i].setCell(cell)

        # C++版の import(G1) に対応
        particles[i]._import(G1)
        particles[i].timestep = 0
        particles[i].setSPHProperties_vec(mu_vec, c1_vec, sph_h)
        particles[i].setupCutoffBlock(sph_h, False)

        # putTMPselected 相当
        particles[i].setupSelectedTMP(N1, N - N1, 0, N1)
        print("particles, moving=", N - N1, " total=", N, file=sys.stderr)

    print("setup done", file=sys.stderr)

    # 未選択粒子の初期配置を出力
    particles[0].putUnSelected(dump_file)

    # 初期の Block ID と density を計算
    for i in range(ndev):
        particles.setGPU(i)
        particles[i].calcBlockID()
        particles[i].calcDensity()

    # ブロックレンジ設定（マルチGPU用）
    for i in range(ndev):
        particles.setGPU(i)
        particles[i].setBlockRange(particles[i].numBlocks(), ndev, i)


def run(params, log_metrics, serialization_file, dump_file):

    deltaT = params["deltaT"]
    stepmax_time = params["stepmax"]
    intaval_time = params["intaval"]

    stepmax = int(stepmax_time / deltaT)
    intaval = max(1, int(intaval_time / deltaT))

    param_g = params["param_g"]

    particles[0].getSelectedTypeID()
    particles[0].getSelectedPosition()
    particles[0].openTMP(dump_file)
    particles[0].putTMPtoO()
    particles[0].waitPutTMP()

    initstep = particles[0].timestep

    ndev = particles.nDevices()

    # 初期速度計算
    for i in range(ndev):
        particles.setGPU(i)
        particles[i].calcVinit(deltaT)

    start_time = time.time()

    for j in range(stepmax):
        if j % 50 == 0:
            print(j, end=" ", file=sys.stderr)
            t_now = j * deltaT
            log_metrics({
                "elapsed_time": time.time() - start_time,
                "time": t_now,
                "delta_t": deltaT,
            }, step=j)

        # density 計算
        for i in range(ndev):
            particles.setGPU(i)
            particles[i].calcBlockID()
            particles[i].getExchangePidRange1()
            particles[i].calcKernels()
            particles[i].calcDensity(True)

        # マルチGPU時の密度交換（必要なら有効化）
        # if ndev > 1:
        #     particles.exchangeForceSelected(ExchangeMode_density)

        for i in range(ndev):
            particles.setGPU(i)
            particles[i].calcDensityPost(True)
            particles[i].selectBlocks()
            particles[i].setSelectedRange(particles[i].numSelectedBlocks(), ndev, i)
            particles[i].calcForce()
            particles[i].calcAcceleration(True)
            particles[i].getExchangePidRange2()

        # マルチGPU時の加速度交換（必要なら有効化）
        # if ndev > 1:
        #     particles.exchangeForceSelected(ExchangeMode_acceleration)

        for i in range(ndev):
            particles.setGPU(i)
            particles[i].RestoreAcceleration()
            particles[i].addAccelerationZ(-param_g)
            particles[i].TimeEvolution(deltaT)
            particles[i].treatAbsoluteCondition()

        if (j + 1) % intaval == 0:
            t_out = (j + 1) * deltaT
            print("\n({}) ".format(t_out), file=sys.stderr)
            particles[0].timestep = j + 1 + initstep
            particles[0].getSelectedPosition()
            particles[0].putTMPtoO()

    particles[0].waitPutTMP()

    # 最終状態ログ
    t_final = stepmax * deltaT
    log_metrics({
        "elapsed_time": time.time() - start_time,
        "time": t_final,
        "delta_t": deltaT,
    }, step=stepmax)

    particles.writeSerialization(serialization_file)
    print("Done.")
