import os
import math
import numpy as np
import matplotlib.pyplot as plt
import time
from pypower.api import runpf, ppoption, makeYbus
from pypower.idx_bus import VM, VA, PD, QD, BS
from pypower.idx_gen import PG, QG, GEN_BUS
from pypower.ext2int import ext2int

from ode_trapezoid import ode_trapezoid

import scienceplots
plt.style.use(['science', 'no-latex'])


def check_initial_consistency(dae_func, x0, mass_diag, t0=0.0):
    """
    Check if initial conditions satisfy the DAE constraints.
    
    Parameters
    ----------
    dae_func : callable
        The DAE function f(t, x)
    x0 : array
        Initial state vector
    mass_diag : array
        Mass matrix diagonal
    t0 : float
        Initial time
    """
    print("Checking initial condition consistency...")
    
    # Evaluate residual at initial time
    f0 = dae_func(t0, x0)
    
    diff_mask = (mass_diag != 0.0)
    alg_mask = (mass_diag == 0.0)
    
    if np.any(diff_mask):
        diff_residual = f0[diff_mask]
        print(f"Differential equation residuals:")
        print(f"  Norm: {np.linalg.norm(diff_residual):.2e}")
        print(f"  Max:  {np.max(np.abs(diff_residual)):.2e}")
        
        # Print worst differential residual
        worst_diff_idx = np.argmax(np.abs(diff_residual))
        global_diff_idx = np.where(diff_mask)[0][worst_diff_idx]
        print(f"  Worst differential residual: index {global_diff_idx}, value {diff_residual[worst_diff_idx]:.2e}")
    
    if np.any(alg_mask):
        alg_residual = f0[alg_mask]
        print(f"Algebraic equation residuals:")
        print(f"  Norm: {np.linalg.norm(alg_residual):.2e}")
        print(f"  Max:  {np.max(np.abs(alg_residual)):.2e}")
        
        # Print worst algebraic residual
        worst_alg_idx = np.argmax(np.abs(alg_residual))
        global_alg_idx = np.where(alg_mask)[0][worst_alg_idx]
        print(f"  Worst algebraic residual: index {global_alg_idx}, value {alg_residual[worst_alg_idx]:.2e}")
    
    overall_norm = np.linalg.norm(f0)
    print(f"Overall residual norm: {overall_norm:.2e}")
    
    if overall_norm > 1e-8:
        print("WARNING: Initial conditions may not be consistent with DAE constraints!")
    else:
        print("Initial conditions appear consistent.")
    
    return f0

def case_ieee9():
    """Base (pre/post-lineOutage) IEEE 9-bus case."""
    bus = np.array([
        [1, 3,   0,   0, 0,   0, 1, 1.04,  0, 345, 1, 1.1, 0.9],
        [2, 2,   0,   0, 0,   0, 1, 1.025, 0, 345, 1, 1.1, 0.9],
        [3, 2,   0,   0, 0,   0, 1, 1.025, 0, 345, 1, 1.1, 0.9],
        [4, 1,   0,   0, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [5, 1, 125,  50, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [6, 1,  90,  30, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [7, 1,   0,   0, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [8, 1, 100,  35, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [9, 1,   0,   0, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
    ], dtype=float)

    gen = np.array([
        [1,   0,   0, 300, -300, 1.040, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 163,   0, 300, -300, 1.025, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3,  85,   0, 300, -300, 1.025, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)

    branch = np.array([
        [1, 4, 0.0,   0.0576, 0.0,   250, 250, 250, 0, 0, 1, -360, 360],
        [4, 6, 0.017, 0.0920, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
        [6, 9, 0.039, 0.1700, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 9, 0.0,   0.0586, 0.0,   300, 300, 300, 0, 0, 1, -360, 360],
        [8, 9, 0.0119,0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085,0.0720, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 7, 0.0,   0.0625, 0.0,   250, 250, 250, 0, 0, 1, -360, 360],
        [5, 7, 0.032, 0.1610, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.01,  0.0850, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
    ], dtype=float)

    return dict(baseMVA=100.0, bus=bus, gen=gen, branch=branch)


def case_ieee9_lineOutage():
    """lineOutageed network: one branch heavily impeded."""
    bus = np.array([
        [1, 3,   0,   0, 0,   0, 1, 1.04,  0, 345, 1, 1.1, 0.9],
        [2, 2,   0,   0, 0,   0, 1, 1.025, 0, 345, 1, 1.1, 0.9],
        [3, 2,   0,   0, 0,   0, 1, 1.025, 0, 345, 1, 1.1, 0.9],
        [4, 1,   0,   0, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [5, 1, 125,  50, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [6, 1,  90,  30, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [7, 1,   0,   0, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [8, 1, 100,  35, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
        [9, 1,   0,   0, 0,   0, 1, 1.0,   0, 345, 1, 1.1, 0.9],
    ], dtype=float)

    gen = np.array([
        [1,   0,   0, 300, -300, 1.040, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 163,   0, 300, -300, 1.025, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3,  85,   0, 300, -300, 1.025, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=float)

    branch = np.array([
        [1, 4, 0.0,     0.0576, 0.0,   250, 250, 250, 0, 0, 1, -360, 360],
        [4, 6, 1000.017, 0.0920, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],  # 'lineOutageed'
        [6, 9, 0.039,   0.1700, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 9, 0.0,     0.0586, 0.0,   300, 300, 300, 0, 0, 1, -360, 360],
        [8, 9, 0.0119,  0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085,  0.0720, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 7, 0.0,     0.0625, 0.0,   250, 250, 250, 0, 0, 1, -360, 360],
        [5, 7, 0.032,   0.1610, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.01,    0.0850, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
    ], dtype=float)

    return dict(baseMVA=100.0, bus=bus, gen=gen, branch=branch)


class Params:
    """Container for all dynamic parameters/constants."""
    pass


def build_params_from_pf(res_base, res_lineOutage, ws=2*math.pi*60.0, new_base_mva=100.0):
    # Extract from PF results
    bus1 = res_base['bus']
    gen1 = res_base['gen']
    branch1 = res_base['branch']
    baseMVA = res_base['baseMVA']

    bus2 = res_lineOutage['bus']
    branch2 = res_lineOutage['branch']

    # Counts
    m = gen1.shape[0]
    n = bus1.shape[0]

    # Ybus (dense arrays)
    ppci1 = ext2int(res_base)
    Ybus1, _, _ = makeYbus(ppci1['baseMVA'], ppci1['bus'], ppci1['branch'])
    ppci2 = ext2int(res_lineOutage)
    Ybus2, _, _ = makeYbus(ppci2['baseMVA'], ppci2['bus'], ppci2['branch'])

    Ybus1 = Ybus1.toarray()
    Ybus2 = Ybus2.toarray()

    # Initial conditions (match MATLAB)
    Vm = bus1[:, VM]
    Va_deg = bus1[:, VA]
    Va = np.deg2rad(Va_deg)

    # Map generator injections to first m buses (IEEE9 ordering)
    # PG_bus, QG_bus arrays per bus (length n)
    PG_bus = np.zeros(n)
    QG_bus = np.zeros(n)
    gen_buses = gen1[:, GEN_BUS].astype(int) - 1  # PYPOWER uses 1-based bus IDs in data
    for i in range(m):
        b = gen_buses[i]
        PG_bus[b] += gen1[i, PG]
        QG_bus[b] += gen1[i, QG]

    # Add shunt susceptance Bs into QG as in MATLAB code (bus col 6)
    QG_bus = QG_bus + bus1[:, BS]

    Pd = bus1[:, PD]
    Qd = bus1[:, QD]

    # pu values - rename to avoid conflict with imported PG constant
    PG_pu = PG_bus / baseMVA
    QG_pu = QG_bus / baseMVA
    PL = Pd / baseMVA
    QL = Qd / baseMVA

    TH0 = Va.copy()
    V0 = Vm.copy()

    # Dynamic parameters
    MD = np.array([
        [23.640, 6.4000, 3.0100],   # H
        [0.1460, 0.8958, 1.3125],   # Xd
        [0.0608, 0.1198, 0.1813],   # Xdp
        [0.0489, 0.0881, 0.1133],   # Xdpp
        [0.0969, 0.8645, 1.2578],   # Xq
        [0.0969, 0.1969, 0.2500],   # Xqp
        [0.0396, 0.0887, 0.0833],   # Xqpp
        [8.9600, 6.0000, 5.8900],   # Td0p
        [0.1150, 0.0337, 0.0420],   # Td0pp
        [0.3100, 0.5350, 0.6000],   # Tq0p
        [0.0330, 0.0780, 0.1875],   # Tq0pp
        [0.0041, 0.0026, 0.0035],   # Rs
        [0.1200, 0.1020, 0.0750],   # Xls
        [0.1 * (2*23.64)/ws, 0.2 * (2*6.4)/ws, 0.3 * (2*3.01)/ws]  # Dm
    ], dtype=float)

    ED = np.vstack([
        20.0 * np.ones((1, m)),      # KA
        0.2  * np.ones((1, m)),      # TA
        1.0  * np.ones((1, m)),      # KE
        0.314* np.ones((1, m)),      # TE
        0.063* np.ones((1, m)),      # KF
        0.350* np.ones((1, m)),      # TF
        0.0039*np.ones((1, m)),      # Ax
        1.555 *np.ones((1, m))       # Bx
    ]).astype(float)

    TD = np.vstack([
        0.10*np.ones((1, m)),        # TCH
        0.05*np.ones((1, m)),        # TSV
        0.05*np.ones((1, m))         # RD
    ]).astype(float)

    p = Params()
    p.ws = ws
    p.baseMVA = baseMVA
    p.m = m
    p.n = n

    (p.H, p.Xd, p.Xdp, p.Xdpp, p.Xq, p.Xqp, p.Xqpp,
     p.Td0p, p.Td0pp, p.Tq0p, p.Tq0pp, p.Rs, p.Xls, p.Dm) = [MD[i, :] for i in range(MD.shape[0])]

    p.KA, p.TA, p.KE, p.TE, p.KF, p.TF, p.Ax, p.Bx = [ED[i, :] for i in range(ED.shape[0])]
    p.TCH, p.TSV, p.RD = [TD[i, :] for i in range(TD.shape[0])]
    p.MH = 2.0 * p.H / p.ws

    p.PL = PL
    p.QL = QL

    # Ybus abs/angle
    p.Yabs = np.abs(Ybus1)
    p.Yang = np.angle(Ybus1)
    p.Yabsf = np.abs(Ybus2)
    p.Yangf = np.angle(Ybus2)

    # Initial generator bus sets (assumes buses 1..m are gens in this case)
    VG0 = V0[:m]
    THG0 = TH0[:m]

    # Initial conditions (match MATLAB)
    Vphasor = VG0 * np.exp(1j * THG0)
    Iphasor = np.conj((PG_pu[:m] + 1j * QG_pu[:m]) / Vphasor)
    E0 = Vphasor + (p.Rs + 1j * p.Xq) * Iphasor
    D0 = np.angle(E0)

    # Park transform currents (d/q axes)
    rot = np.exp(-1j * (D0 - np.pi/2.0))
    Id0 = np.real(Iphasor * rot)
    Iq0 = np.imag(Iphasor * rot)

    Edp0 = (p.Xq - p.Xqp) * Iq0
    Si2q0 = (p.Xls - p.Xq) * Iq0
    Eqp0 = p.Rs * Iq0 + p.Xdp * Id0 + V0[:m] * np.cos(D0 - TH0[:m])
    Si1d0 = Eqp0 - (p.Xdp - p.Xls) * Id0
    Efd0 = Eqp0 + (p.Xd - p.Xdp) * Id0

    TM0 = ((p.Xdpp - p.Xls)/(p.Xdp - p.Xls)) * Eqp0*Iq0 \
        + ((p.Xdp  - p.Xdpp)/(p.Xdp - p.Xls)) * Si1d0*Iq0 \
        + ((p.Xqpp - p.Xls)/(p.Xqp - p.Xls)) * Edp0*Id0 \
        - ((p.Xqp  - p.Xqpp)/(p.Xqp - p.Xls)) * Si2q0*Id0 \
        + (p.Xqpp - p.Xdpp) * Id0 * Iq0

    VR0 = (p.KE + p.Ax * np.exp(p.Bx * Efd0)) * Efd0
    RF0 = (p.KF / p.TF) * Efd0

    # Vref = VG + VR/KA
    p.Vref = V0[:m] + VR0 / p.KA
    # Prime mover
    PSV0 = TM0
    p.PC = PSV0.copy()

    # Store fixed initial bus voltages/angles as references
    p.V0 = V0.copy()
    p.TH0 = TH0.copy()

    # Assemble initial state vector
    x0 = np.zeros(11*m, dtype=float)
    x0[0*m:1*m] = Eqp0
    x0[1*m:2*m] = Si1d0
    x0[2*m:3*m] = Edp0
    x0[3*m:4*m] = Si2q0
    x0[4*m:5*m] = D0
    x0[5*m:6*m] = p.ws * np.ones(m)  # w
    x0[6*m:7*m] = Efd0
    x0[7*m:8*m] = RF0
    x0[8*m:9*m] = VR0
    x0[9*m:10*m] = TM0
    x0[10*m:11*m] = PSV0

    # Algebraic: Id, Iq, V, TH
    a0 = np.concatenate([Id0, Iq0, V0, TH0])

    x01 = np.concatenate([x0, a0])

    # Mass diagonal: ones for differential (11*m), zeros for algebraic (2*m + 2*n)
    mass_diag = np.concatenate([np.ones(11*m, dtype=float), np.zeros(2*m + 2*n, dtype=float)])

    return p, x01, mass_diag


def make_dae_func(p: Params, Yabs, Yang, voltage_dependent_loads=True):
    m, n = p.m, p.n

    def f(t, x):
        # Unpack
        Eqp = x[0*m:1*m]
        Si1d = x[1*m:2*m]
        Edp = x[2*m:3*m]
        Si2q = x[3*m:4*m]
        Delta = x[4*m:5*m]
        w = x[5*m:6*m]
        Efd = x[6*m:7*m]
        RF = x[7*m:8*m]
        VR = x[8*m:9*m]
        TM = x[9*m:10*m]
        PSV = x[10*m:11*m]
        Id = x[11*m:12*m]
        Iq = x[12*m:13*m]
        V = x[13*m:13*m+n]
        TH = x[13*m+n:13*m+2*n]

        # COI
        COI = np.sum(w * p.MH) / np.sum(p.MH)

        # Loads
        if voltage_dependent_loads:
            # (V/V0)^alpha, (V/V0)^beta
            ratio = np.maximum(V / p.V0, 1e-9)
            PL2 = p.PL * (ratio ** 2.0)  # alpha=2
            QL2 = p.QL * (ratio ** 2.0)  # beta = alpha
        else:
            PL2 = p.PL
            QL2 = p.QL

        VG = V[:m]
        THG = TH[:m]
        angle_diff = Delta - THG

        # Differential equations (S1..S11)
        S1 = (1.0/p.Td0p) * (-Eqp - (p.Xd - p.Xdp) * (Id - ((p.Xdp - p.Xdpp) / (p.Xdp - p.Xls)**2) * (Si1d + (p.Xdp - p.Xls)*Id - Eqp)) + Efd)
        S2 = (1.0/p.Td0pp) * (-Si1d + Eqp - (p.Xdp - p.Xls)*Id)
        S3 = (1.0/p.Tq0p) * (-Edp + (p.Xq - p.Xqp) * (Iq - ((p.Xqp - p.Xqpp) / (p.Xqp - p.Xls)**2) * (Si2q + (p.Xqp - p.Xls)*Iq + Edp)))
        S4 = (1.0/p.Tq0pp) * (-Si2q - Edp - (p.Xqp - p.Xls)*Iq)
        S5 = w - COI
        S6 = (p.ws/(2.0*p.H)) * (TM
            - ((p.Xdpp - p.Xls)/(p.Xdp - p.Xls)) * Eqp*Iq
            - ((p.Xdp  - p.Xdpp)/(p.Xdp - p.Xls)) * Si1d*Iq
            - ((p.Xqpp - p.Xls)/(p.Xqp - p.Xls)) * Edp*Id
            + ((p.Xqp  - p.Xqpp)/(p.Xqp - p.Xls)) * Si2q*Id
            - (p.Xqpp - p.Xdpp) * Id*Iq
            - p.Dm * (w - p.ws))

        S7 = (1.0/p.TE) * (-(p.KE + p.Ax * np.exp(p.Bx * Efd)) * Efd + VR)
        S8 = (1.0/p.TF) * (-RF + (p.KF/p.TF) * Efd)
        S9 = (1.0/p.TA) * (-VR + p.KA*RF - (p.KA*p.KF/p.TF)*Efd + p.KA*(p.Vref - VG))
        S10 = (1.0/p.TCH) * (-TM + PSV)
        S11 = (1.0/p.TSV) * (-PSV + p.PC - (1.0/p.RD) * (w/p.ws - 1.0))

        # Network sums (generator buses)
        ang1 = TH[:m, None] - TH[None, :] - Yang[:m, :]
        mag1 = V[:m, None] * V[None, :] * Yabs[:m, :]
        sum1 = np.sum(mag1 * np.cos(ang1), axis=1)
        sum2 = np.sum(mag1 * np.sin(ang1), axis=1)

        SE1 = p.Rs*Id - p.Xqpp*Iq - ((p.Xqpp - p.Xls)/(p.Xqp - p.Xls)) * Edp + ((p.Xqp - p.Xqpp)/(p.Xqp - p.Xls)) * Si2q + VG * np.sin(angle_diff)
        SE2 = p.Rs*Iq + p.Xdpp*Id - ((p.Xdpp - p.Xls)/(p.Xdp - p.Xls)) * Eqp - ((p.Xdp - p.Xdpp)/(p.Xdp - p.Xls)) * Si1d + VG * np.cos(angle_diff)

        PV1 = (Id * VG * np.sin(angle_diff) + Iq * VG * np.cos(angle_diff)) - PL2[:m] - sum1
        PV2 = (Id * VG * np.cos(angle_diff) - Iq * VG * np.sin(angle_diff)) - QL2[:m] - sum2

        # Non-generator buses
        ang2 = TH[m:, None] - TH[None, :] - Yang[m:, :]
        mag2 = V[m:, None] * V[None, :] * Yabs[m:, :]
        sum3 = np.sum(mag2 * np.cos(ang2), axis=1)
        sum4 = np.sum(mag2 * np.sin(ang2), axis=1)

        PQ1 = -PL2[m:] - sum3
        PQ2 = -QL2[m:] - sum4

        # Stack
        out = np.concatenate([
            S1, S2, S3, S4, S5, S6, S7, S8, S9, S10, S11,
            SE1, SE2, PV1, PQ1, PV2, PQ2
        ])
        return out.astype(float)

    return f


def main():
    # Config
    ws = 2*math.pi*60.0

    # Stages
    st1 = 1e-4  # pre-lineOutage
    st2 = 1e-4  # lineOutage-on
    st3 = 1e-4  # post-lineOutage
    # st1 = 5e-5  # pre-lineOutage
    # st2 = 5e-5  # lineOutage-on
    # st3 = 5e-5  # post-lineOutage

    tf1 = 0.1
    tf2 = 0.9
    tfinal = 1.0

    # Prepare PF options
    mpopt = ppoption(PF_ALG=1, ENFORCE_Q_LIMS=0, VERBOSE=0, OUT_ALL=0)

    # Base and lineOutage cases
    mpc_base = case_ieee9()
    mpc_lineOutage = case_ieee9_lineOutage()

    # Run PF
    res1, success1 = runpf(mpc_base, mpopt)
    if not success1:
        raise RuntimeError("Base case power flow did not converge.")
    res2, success2 = runpf(mpc_lineOutage, mpopt)
    if not success2:
        raise RuntimeError("lineOutage case power flow did not converge.")

    # Parameters and initial conditions
    p, x01, mass_diag = build_params_from_pf(res1, res2, ws=ws, new_base_mva=mpc_base['baseMVA'])

    # Build stage-specific DAEs
    dae_pre = make_dae_func(p, p.Yabs, p.Yang, voltage_dependent_loads=False)
    dae_lineOutage = make_dae_func(p, p.Yabsf, p.Yangf, voltage_dependent_loads=False)
    dae_post = make_dae_func(p, p.Yabs, p.Yang, voltage_dependent_loads=False)

    # Check initial conditions (optional)
    check_initial_consistency(dae_pre, x01, mass_diag)


    # Time grids
    t1 = np.arange(0.0, tf1 + 1e-12, st1)
    if t1[-1] < tf1: t1 = np.append(t1, tf1)

    t2_local = np.arange(0.0, (tf2 - tf1) + 1e-12, st2)
    if t2_local[-1] < (tf2 - tf1): t2_local = np.append(t2_local, (tf2 - tf1))

    t3_local = np.arange(0.0, (tfinal - tf2) + 1e-12, st3)
    if t3_local[-1] < (tfinal - tf2): t3_local = np.append(t3_local, (tfinal - tf2))


#### start timing for tts (time to solution)
    start_time_tts = time.time()

    # Solve Stage 1
    t_s1, X_s1, it_t1, it_c1 = ode_trapezoid(
        dae_pre, t1, x01, mass_diag=mass_diag, max_step=None,
        progress=True, progress_interval=1.0, newton_tol=1e-6, newton_max=1000
    )

    # Stage 2 initial
    x02 = X_s1[-1, :].copy()

    # Solve Stage 2
    t_s2_local, X_s2, it_t2_local, it_c2 = ode_trapezoid(
        dae_lineOutage, t2_local, x02, mass_diag=mass_diag, max_step=None,
        progress=True, progress_interval=1.0, newton_tol=1e-6, newton_max=1000
    )
    t_s2 = t_s2_local + tf1
    it_t2 = it_t2_local + tf1

    # Stage 3 initial
    x03 = X_s2[-1, :].copy()

    # Solve Stage 3
    t_s3_local, X_s3, it_t3_local, it_c3 = ode_trapezoid(
        dae_post, t3_local, x03, mass_diag=mass_diag, max_step=None,
        progress=True, progress_interval=1.0, newton_tol=1e-6, newton_max=1000
    )
    t_s3 = t_s3_local + tf2
    it_t3 = it_t3_local + tf2

    # Merge
    t = np.concatenate([t_s1, t_s2[1:], t_s3[1:]])
    X = np.vstack([X_s1, X_s2[1:, :], X_s3[1:, :]])

    end_time_tts = time.time()

#### end timing for tts (time to solution)
    tts_time = end_time_tts - start_time_tts
    print(f"\nTotal solve time: {tts_time:.4f} seconds")

    # Extract variables
    m = p.m
    n = p.n
    Eqp = X[:, 0*m:1*m]
    Si1d = X[:, 1*m:2*m]
    Edp = X[:, 2*m:3*m]
    Si2q = X[:, 3*m:4*m]
    Delta = X[:, 4*m:5*m]
    w = X[:, 5*m:6*m]
    Efd = X[:, 6*m:7*m]
    RF = X[:, 7*m:8*m]
    VR = X[:, 8*m:9*m]
    TM = X[:, 9*m:10*m]
    PSV = X[:, 10*m:11*m]
    Id = X[:, 11*m:12*m]
    Iq = X[:, 12*m:13*m]
    V = X[:, 13*m:13*m+n]
    TH = X[:, 13*m+n:13*m+2*n]

    # Plot frequency
    freq = w / (2.0 * math.pi)
    plt.figure(figsize=(8, 4))
    for gi in range(m):
        plt.plot(t, freq[:, gi], linewidth=2, label=f"G{gi+1}")
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.legend(loc="best")
    plt.tight_layout()

    # Output naming
    # dtLabel = f"dt1_{st1}_dt2_{st2}_dt3_{st3}".replace('.', '_')
    # ode_solver = "TR"
    eventLabel = f"tF_{tf1}_{tf2}".replace('.', '_')
    # runLabel = f"IEEE9Bus_lineOutage_{eventLabel}_dt_{st1}_{ode_solver}"
    runLabel = f"IEEE9Bus_lineOutage_{eventLabel}_dt_{st1}"
    outPrefix = f"event_{runLabel}"

    figFile = f"{outPrefix}.png"
    plt.savefig(figFile, dpi=150)

    # Plot voltage at bus 6
    plt.figure(figsize=(8, 4))
    bus6_idx = 5  # Bus 6 is at index 5 (0-based indexing)
    plt.plot(t, V[:, bus6_idx], linewidth=2, label='Bus 6 Voltage')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage at Bus 6")
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save bus 6 voltage plot
    bus6_figFile = f"{outPrefix}_bus6_voltage.png"
    plt.savefig(bus6_figFile, dpi=150)

    # Plot voltage at bus 6 - zoomed view (0.08 - 0.2)
    plt.figure(figsize=(8, 4))
    plt.plot(t, V[:, bus6_idx], linewidth=2, label='Bus 6 Voltage')
    plt.grid(True)
    plt.xlabel("Time (s)")
    plt.ylabel("Voltage (p.u.)")
    plt.title("Voltage at Bus 6 (Zoomed: 0.08 - 0.2 p.u.)")
    plt.xlim(0.08, 0.2)  # Set y-axis limits to zoom range
    plt.ylim(0.9, 1.02)  # Set y-axis limits to zoom range
    plt.legend(loc="best")
    plt.tight_layout()
    
    # Save zoomed bus 6 voltage plot
    bus6_zoomed_figFile = f"{outPrefix}_bus6_voltage_zoomed.png"
    plt.savefig(bus6_zoomed_figFile, dpi=150)

    # CSVs
    # 1) time + all generator speeds w
    dataToSave = np.column_stack([t, w])
    np.savetxt(f"{outPrefix}.csv", dataToSave, delimiter=',', fmt="%.10g")

    # 2) time + bus voltages
    dataToSave2 = np.column_stack([t, V])
    np.savetxt(f"{outPrefix}_V_.csv", dataToSave2, delimiter=',', fmt="%.10g")

    # 3) Newton/root iteration counts
    iter_t_global = np.concatenate([t_s1[1:] if X_s1.shape[0] > 1 else np.array([]),
                                    it_t2, it_t3])
    iter_counts_global = np.concatenate([it_c1 if X_s1.shape[0] > 1 else np.array([]),
                                          it_c2, it_c3])
    iterData = np.column_stack([iter_t_global, iter_counts_global])
    np.savetxt(f"{outPrefix}_newton_iterations.csv", iterData, delimiter=',', fmt="%.10g")

    print(f"\nSaved: {figFile}, {outPrefix}.csv, {outPrefix}_V_.csv, {outPrefix}_newton_iterations.csv")


if __name__ == "__main__":
    main()