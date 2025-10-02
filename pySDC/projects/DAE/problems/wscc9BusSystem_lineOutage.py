import numpy as np
from pySDC.projects.DAE.misc.problemDAE import ProblemDAE
from pySDC.core.errors import ParameterError
from pypower.makeYbus import makeYbus
from pypower.ext2int import ext2int
from pypower.api import runpf, ppoption


def ieee9_case_raw():
    """
    Returns the unsolved (base) IEEE 9-bus MATPOWER-style dictionary (matching MATLAB IEEE9Bus function).
    """
    ppc = {}
    ppc['baseMVA'] = 100.0
    ppc['bus'] = np.array([
        [1, 3,   0,   0, 0, 0, 1, 1.04,  0.0, 345, 1, 1.1, 0.9],
        [2, 2,   0,   0, 0, 0, 1, 1.025, 0.0, 345, 1, 1.1, 0.9],
        [3, 2,   0,   0, 0, 0, 1, 1.025, 0.0, 345, 1, 1.1, 0.9],
        [4, 1,   0,   0, 0, 0, 1, 1.00,  0.0, 345, 1, 1.1, 0.9],
        [5, 1, 125,  50, 0, 0, 1, 1.00,  0.0, 345, 1, 1.1, 0.9],
        [6, 1,  90,  30, 0, 0, 1, 1.00,  0.0, 345, 1, 1.1, 0.9],
        [7, 1,   0,   0, 0, 0, 1, 1.00,  0.0, 345, 1, 1.1, 0.9],
        [8, 1, 100,  35, 0, 0, 1, 1.00,  0.0, 345, 1, 1.1, 0.9],
        [9, 1,   0,   0, 0, 0, 1, 1.00,  0.0, 345, 1, 1.1, 0.9],
    ])
    ppc['gen'] = np.array([
        [1,   0,  0, 300, -300, 1.040, 100, 1, 250, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [2, 163,  0, 300, -300, 1.025, 100, 1, 300, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [3,  85,  0, 300, -300, 1.025, 100, 1, 270, 10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    ppc['branch'] = np.array([
        [1, 4, 0.0,   0.0576, 0.0,   250, 250, 250, 0, 0, 1, -360, 360],
        [4, 6, 0.017, 0.0920, 0.158, 250, 250, 250, 0, 0, 1, -360, 360],
        [6, 9, 0.039, 0.1700, 0.358, 150, 150, 150, 0, 0, 1, -360, 360],
        [3, 9, 0.0,   0.0586, 0.0,   300, 300, 300, 0, 0, 1, -360, 360],
        [8, 9, 0.0119,0.1008, 0.209, 150, 150, 150, 0, 0, 1, -360, 360],
        [7, 8, 0.0085,0.0720, 0.149, 250, 250, 250, 0, 0, 1, -360, 360],
        [2, 7, 0.0,   0.0625, 0.0,   250, 250, 250, 0, 0, 1, -360, 360],
        [5, 7, 0.032, 0.1610, 0.306, 250, 250, 250, 0, 0, 1, -360, 360],
        [4, 5, 0.01,  0.0850, 0.176, 250, 250, 250, 0, 0, 1, -360, 360],
    ])
    return ppc

def WSCC9Bus(run_power_flow=True):
    """
    Build (optionally) solved WSCC/IEEE 9-bus case mimicking MATLAB script:
    - run power flow
    - return solved bus/gen/branch with baseMVA
    """
    ppc = ieee9_case_raw()
    if not run_power_flow:
        return ppc
    mpopt = ppoption(PF_ALG=1, ENFORCE_Q_LIMS=0, VERBOSE=0, OUT_ALL=0)
    results, success = runpf(ppc, mpopt)
    if not success:
        raise RuntimeError("Power flow did not converge for IEEE 9-bus case.")
    return {
        'baseMVA': results['baseMVA'],
        'bus': results['bus'],
        'branch': results['branch'],
        'gen': results['gen'],
    }


def get_initial_Ybus():
    """
    Builds the initial Y bus using makeYbus from pypower.
    
    Returns
    -------
    ybus : np.2darray
        The Y bus matrix for the initial system configuration.
    """
    ppc = WSCC9Bus()
    ppci = ext2int(ppc)
    ybus, _, _ = makeYbus(ppci['baseMVA'], ppci['bus'], ppci['branch'])
    # Force dense complex128 ndarray
    ybus = ybus.toarray().astype(np.complex128)

    # print(ybus) # debug
    return ybus


def get_event_Ybus():
    """
    Builds the Y bus for the FAULT (modified line 4-6 impedance) scenario using makeYbus from pypower.
    In the MATLAB files (IEEE9Bus2.m) the line between buses 4 and 6 is NOT removed; instead its
    series r and x are drastically increased:
        normal: r = 0.017, x = 0.092
        fault : r = 1.017, x = 1.092
    """
    ppc = WSCC9Bus()
    ppc_event = ppc.copy()
    # Locate the (4,6) branch (second row in current ordering) and modify r, x
    # Original ordering in ieee9_case_raw():
    # [1,4], [4,6], [6,9], [3,9], [8,9], [7,8], [2,7], [5,7], [4,5]
    # So index 1 corresponds to branch 4-6
    fault_idx = None
    for k, br in enumerate(ppc_event['branch']):
        if int(br[0]) == 4 and int(br[1]) == 6:
            fault_idx = k
            break
    if fault_idx is None:
        raise RuntimeError("Branch 4-6 not found for fault modification.")
    ppc_event['branch'][fault_idx, 2] = 1000.017  # r
    # ppc_event['branch'][fault_idx, 3] = 10.092  # x
    # Rebuild Ybus
    ppci = ext2int(ppc_event)
    ybus, _, _ = makeYbus(ppci['baseMVA'], ppci['bus'], ppci['branch'])
    ybus = ybus.toarray().astype(np.complex128)
    return ybus


class WSCC9BusSystem(ProblemDAE):
    r"""
    Example implementing the WSCC 9 Bus system [1]_. For this complex model, the equations can be found in [2]_, and
    sub-transient and turbine parameters are taken from [3]_. The network data of the system are taken from MATPOWER [4]_.

    Parameters
    ----------
    nvars : int
        Number of unknowns of the system of DAEs (not used here, since it is set up inside this class).
    newton_tol : float
        Tolerance for Newton solver.

    Attributes
    ----------
    mpc : dict
        Contains the data for the buses, branches, generators, and the Ybus.
    m : int
        Number of machines used in the network.
    n : int
        Number of buses used in the network.
    baseMVA : float
        Base value of the apparent power.
    ws : float
        Generator synchronous speed in rad per second.
    ws_vector : np.1darray
        Vector containing ``ws``.
    MD : np.2darray
        Machine data.
    ED : np.2darray
        Excitation data.
    TD : np.2darray
        Turbine data.
    bus : np.2darray
        Data for the buses.
    branch : np.2darray
        Data for the branches in the power system.
    gen : np.2darray
        Data for generators in the system.
    Ybus : np.2darray
        Ybus.
    YBus_line6_8_outage : np.2darray
        Contains the data for the line outage in the power system, where line at bus6 is outaged.
    psv_max : float
         Maximum valve position.
    IC1 : list
        Contains the :math:`8`-th row of the ``bus`` matrix.
    IC2 : list
        Contains the :math:`9`-th row of the ``bus`` matrix.
    IC3 : list
        Generator values divided by ``baseMVA``.
    IC4 : list
        Generator values divided by ``baseMVA``.
    IC5 : list
        Loads divided by ``baseMVA``.
    IC6 : list
        Loads divided by ``baseMVA``.
    genP : list
        Contains data for one generator from the ``mpc``.
    IC : list
        Contains all the values of `IC1, IC2, IC3, IC4, IC5, IC6`.
    PL : list
        Contains the :math:`5`-th column of ``IC``.
    QL : list
        Contains the :math:`6`-th column of ``IC``.
    PG : np.1darray
        Contains the :math:`3`-rd column of ``IC``.
    QG : np.1darray
        Contains the :math:`4`-th column of ``IC``.
    TH0 : np.1darray
        Initial condition for angle of bus voltage in rad.
    V0 : np.1darray
        Contains the :math:`1`-st column of ``IC``, initial condition for magnitude of bus voltage in per unit.
    VG0 : np.1darray
        Initial condition for complex voltage phasor.
    THG0 : np.1darray
        Initial condition for angle of the bus voltage in rad.
    H : np.1darray
        Shaft inertia constant in second.
    Xd : np.1darray
        d-axis reactance in per unit.
    Xdp : np.1darray
        Transient d-axis reactance in per unit.
    Xdpp : np.1darray
        Sub-transient d-axis reactance in per unit.
    Xq : np.1darray
        q-axis reactance in per unit.
    Xqp : np.1darray
        Transient q-axis reactance in per unit.
    Xqpp : np.1darray
        Sub-transient q-axis reactance in per unit.
    Td0p : np.1darray
        d-axis time constant associated with :math:`E_q'` in second.
    Td0pp : np.1darray
        d-axis time constant associated with :math:`\psi_{1d}` in second.
    Tq0p : np.1darray
        q-axis time constant associated with :math:`E_d'` in second.
    Tq0pp : np.1darray
        q-axis time constant associated with :math:`\psi_{2q}` in second.
    Rs : np.1darray
        Stator resistance in per unit.
    Xls : np.1darray
        Parameter :math:`X_{\ell s}`.
    Dm : np.1darray
        Rotor angle in rad.
    KA : np.1darray
        Amplifier gain.
    TA : np.1darray
        Amplifier time constant in second.
    KE : np.1darray
        Separate or self-excited constant.
    TE : np.1darray
        Parameter :math:`T_E`.
    KF : np.1darray
        Parameter _math:`K_F`.
    TF : np.1darray
        Parameter :math:`T_F`.
    Ax : np.1darray
        Constant :math:`A_x` of the saturation function :math:`S_{E_i}`.
    Bx : np.1darray
        Constant :math:`B_x` of the saturation function :math:`S_{E_i}`.
    TCH : np.1darray
        Incremental steam chest time constant in second.
    TSV : np.1darray
        Steam valve time constant in second.
    RD : np.1darray
        Speed regulation quantity in Hz/per unit.
    MH : float
        Factor :math:`\frac{2 H_i}{w_s}`.
    QG : np.1darray
        Used to compute :math:`I_{phasor}`.
    Vphasor : np.1darray
        Complex voltage phasor.
    Iphasor : np.1darray
        Complex current phasor.
    E0 : np.1darray
        Initial internal voltage of the synchronous generator.
    Em : np.1darray
        Absolute values of ``E0``.
    D0 : np.1darray
        Initial condition for rotor angle in rad.
    Id0 : np.1darray
        Initial condition for d-axis current in per unit.
    Iq0 : np.1darray
        Initial condition for q-axis current in per unit.
    Edp0 : np.1darray
        Initial condition for d-axis transient internal voltages in per unit.
    Si2q0 : np.1darray
        Initial condition for damper winding 2q flux linkages in per unit.
    Eqp0 : np.1darray
        Initial condition for q-axis transient internal voltages in per unit.
    Si1d0 : np.1darray
        Initial condition for damper winding 1d flux linkages in per unit.
    Efd0 : np.1darray
        Initial condition for field winding fd flux linkages in per unit.
    TM0 : np.1darray
        Initial condition for mechanical input torque in per unit.
    VR0 : np.1darray
        Initial condition for exciter input in per unit.
    RF0 : np.1darray
        Initial condition for exciter feedback in per unit.
    Vref : np.1darray
        Reference voltage input in per unit.
    PSV0 : np.1darray
        Initial condition for steam valve position in per unit.
    PC : np.1darray
        Initial condition for control power input in per unit.
    alpha : int
        Active load parameter.
    beta : int
        Reactive load parameter.
    bb1, aa1 : list of ndarrays
        Used to access on specific values of ``TH``.
    bb2, aa2 : list of ndarrays
        Used to access on specific values of ``TH``.
    t_switch : float
        Time the event found by detection.
    nswitches : int
        Number of events found by detection.

    References
    ----------
    .. [1] WSCC 9-Bus System - Illinois Center for a Smarter Electric Grid. https://icseg.iti.illinois.edu/wscc-9-bus-system/
    .. [2] P. W. Sauer, M. A. Pai. Power System Dynamics and Analysis. John Wiley & Sons (2008).
    .. [3] I. Abdulrahman. MATLAB-Based Programs for Power System Dynamics Analysis. IEEE Open Access Journal of Power and Energy.
       Vol. 7, No. 1, pp. 59–69 (2020).
    .. [4] R. D. Zimmerman, C. E. Murillo-Sánchez, R. J. Thomas. MATPOWER: Steady-State Operations, Planning, and Analysis Tools
       for Power Systems Research and Education. IEEE Transactions on Power Systems. Vol. 26, No. 1, pp. 12–19 (2011).
    """

    def __init__(self, newton_tol=1e-10):
        """Initialization routine"""
        m, n = 3, 9
        nvars = 11 * m + 2 * m + 2 * n
        # invoke super init, passing number of dofs
        super().__init__(nvars=nvars, newton_tol=newton_tol)
        self._makeAttributeAndRegister('m', 'n', localVars=locals())
        self.mpc = WSCC9Bus()

        self.baseMVA = self.mpc['baseMVA']
        self.ws = 2 * np.pi * 60
        self.ws_vector = self.ws * np.ones(self.m)

        # self.MD = np.array(
        #     [
        #         42.000 * np.ones(self.m),  # 1 - H
        #         0.1000 * np.ones(self.m),  # 2 - Xd
        #         0.0310 * np.ones(self.m),  # 3 - Xdp
        #         0.0250 * np.ones(self.m),  # 4 - Xdpp
        #         0.0690 * np.ones(self.m),  # 5 - Xq
        #         0.0417 * np.ones(self.m),  # 6 - Xqp
        #         0.0250 * np.ones(self.m),  # 7 - Xqpp
        #         10.200 * np.ones(self.m),  # 8 - Td0p
        #         0.0500 * np.ones(self.m),  # 9 - Td0pp
        #         1.5000 * np.ones(self.m),  # 10 - Tq0p
        #         0.0350 * np.ones(self.m),  # 11 - Tq0pp
        #         0.0000 * np.ones(self.m),  # 12 - Rs
        #         0.0125 * np.ones(self.m),  # 13 - Xls
        #         0.1000 * np.ones(self.m),  # 14 - Dm
        #     ]
        # )
        self.MD = np.array([
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
            [0.1 * (2*23.64)/self.ws, 0.2 * (2*6.4)/self.ws, 0.3 * (2*3.01)/self.ws]  # Dm
        ], dtype=float)

        # Excitation data (ED) updated to match MATLAB script
        self.ED = np.array(
            [
                20.0 * np.ones(self.m),  # 1 - KA
                0.2 * np.ones(self.m),  # 2 - TA
                1.0 * np.ones(self.m),  # 3 - KE
                0.314 * np.ones(self.m),  # 4 - TE
                0.063 * np.ones(self.m),  # 5 - KF
                0.350 * np.ones(self.m),  # 6 - TF
                0.0039 * np.ones(self.m),  # 7 - Ax
                1.555 * np.ones(self.m),  # 8 - Bx
            ]
        )

        # Turbine data (TD) updated to match MATLAB script
        self.TD = np.array(
            [
                0.10 * np.ones(self.m),  # 1 - TCH
                0.05 * np.ones(self.m),  # 2 - TSV (changed from 0.05 to 0.10)
                0.05 * np.ones(self.m),  # 3 - RD
            ]
        )

        self.bus = self.mpc['bus']
        self.branch = self.mpc['branch']
        self.gen = self.mpc['gen']
        
        # Build Y bus using makeYbus instead of fixed matrices
        self.YBus = get_initial_Ybus()
        # Fault (modified impedance) network
        self.YBus_fault = get_event_Ybus()

        # excitation limiter vmax
        # self.vmax = 2.1
        self.psv_max = 1.0

        self.IC1 = [row[7] for row in self.bus]  # Column 8 in MATLAB is indexed as 7 in Python (0-based index)
        self.IC2 = [row[8] for row in self.bus]  # Column 9 in MATLAB is indexed as 8 in Python

        n_prev, m_prev = self.n, self.m
        self.n = len(self.bus)  # Number of rows in 'bus' list; self.n already defined above?!
        self.m = len(self.gen)  # Number of rows in 'gen' list; self.m already defined above?!
        if n_prev != 9 or m_prev != 3:
            raise ParameterError("Number of rows in bus or gen not equal to initialised n or m!")

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][1]
        self.genP = gen0
        self.IC3 = [val / self.baseMVA for val in self.genP]

        gen0 = [0] * self.n
        for i in range(self.m):
            gen0[i] = self.gen[i][2]
        genQ = gen0
        for i in range(self.n):
            genQ[i] += self.bus[i][5]  # Column 6 in MATLAB is indexed as 5 in Python
        self.IC4 = [val / self.baseMVA for val in genQ]

        self.IC5 = [row[2] for row in self.bus]  # Column 3 in MATLAB is indexed as 2 in Python
        self.IC5 = [val / self.baseMVA for val in self.IC5]

        self.IC6 = [row[3] for row in self.bus]  # Column 4 in MATLAB is indexed as 3 in Python
        self.IC6 = [val / self.baseMVA for val in self.IC6]

        self.IC = list(zip(self.IC1, self.IC2, self.IC3, self.IC4, self.IC5, self.IC6))

        self.PL = [row[4] for row in self.IC]  # Column 5 in MATLAB is indexed as 4 in Python
        self.QL = [row[5] for row in self.IC]  # Column 6 in MATLAB is indexed as 5 in Python

        self.PG = np.array([row[2] for row in self.IC])  # Column 3 in MATLAB is indexed as 2 in Python
        self.QG = np.array([row[3] for row in self.IC])  # Column 4 in MATLAB is indexed as 3 in Python

        self.TH0 = np.array([row[1] * np.pi / 180 for row in self.IC])
        self.V0 = np.array([row[0] for row in self.IC])
        self.VG0 = self.V0[: self.m]
        self.THG0 = self.TH0[: self.m]

        # Extracting values from the MD array
        self.H = self.MD[0, :]
        self.Xd = self.MD[1, :]
        self.Xdp = self.MD[2, :]
        self.Xdpp = self.MD[3, :]
        self.Xq = self.MD[4, :]
        self.Xqp = self.MD[5, :]
        self.Xqpp = self.MD[6, :]
        self.Td0p = self.MD[7, :]
        self.Td0pp = self.MD[8, :]
        self.Tq0p = self.MD[9, :]
        self.Tq0pp = self.MD[10, :]
        self.Rs = self.MD[11, :]
        self.Xls = self.MD[12, :]
        self.Dm = self.MD[13, :]

        # Extracting values from the ED array
        self.KA = self.ED[0, :]
        self.TA = self.ED[1, :]
        self.KE = self.ED[2, :]
        self.TE = self.ED[3, :]
        self.KF = self.ED[4, :]
        self.TF = self.ED[5, :]
        self.Ax = self.ED[6, :]
        self.Bx = self.ED[7, :]

        # Extracting values from the TD array
        self.TCH = self.TD[0, :]
        self.TSV = self.TD[1, :]
        self.RD = self.TD[2, :]

        # Calculate MH
        self.MH = 2 * self.H / self.ws

        # Represent QG as complex numbers
        self.QG = self.QG.astype(complex)

        # Convert VG0 and THG0 to complex phasors
        self.Vphasor = self.VG0 * np.exp(1j * self.THG0)

        # Calculate Iphasor
        self.Iphasor = np.conj(np.divide(self.PG[:m] + 1j * self.QG[:m], self.Vphasor))

        # Calculate E0
        self.E0 = self.Vphasor + (self.Rs + 1j * self.Xq) * self.Iphasor

        # Calculate Em, D0, Id0, and Iq0
        self.Em = np.abs(self.E0)
        self.D0 = np.angle(self.E0)
        self.Id0 = np.real(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))
        self.Iq0 = np.imag(self.Iphasor * np.exp(-1j * (self.D0 - np.pi / 2)))

        # Calculate Edp0, Si2q0, Eqp0, and Si1d0
        self.Edp0 = (self.Xq - self.Xqp) * self.Iq0
        self.Si2q0 = (self.Xls - self.Xq) * self.Iq0
        self.Eqp0 = self.Rs * self.Iq0 + self.Xdp * self.Id0 + self.V0[: self.m] * np.cos(self.D0 - self.TH0[: self.m])
        self.Si1d0 = self.Eqp0 - (self.Xdp - self.Xls) * self.Id0

        # Calculate Efd0 and TM0
        self.Efd0 = self.Eqp0 + (self.Xd - self.Xdp) * self.Id0
        self.TM0 = (
            ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * self.Eqp0 * self.Iq0
            + ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * self.Si1d0 * self.Iq0
            + ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * self.Edp0 * self.Id0
            - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * self.Si2q0 * self.Id0
            + (self.Xqpp - self.Xdpp) * self.Id0 * self.Iq0
        )

        # Calculate VR0 and RF0
        self.VR0 = (self.KE + self.Ax * np.exp(self.Bx * self.Efd0)) * self.Efd0
        self.RF0 = (self.KF / self.TF) * self.Efd0

        # Calculate Vref and PSV0
        self.Vref = self.V0[: self.m] + self.VR0 / self.KA
        self.PSV0 = self.TM0
        self.PC = self.PSV0

        self.alpha = 2
        self.beta = 2

        # self.bb1, self.aa1 = np.meshgrid(np.arange(0, self.m), np.arange(0, self.n))
        # self.bb1, self.aa1 = self.bb1.astype(int), self.aa1.astype(int)

        # # Create matrix grid to eliminate for-loops (load buses)
        # self.bb2, self.aa2 = np.meshgrid(np.arange(self.m, self.n), np.arange(0, self.n))
        # self.bb2, self.aa2 = self.bb2.astype(int), self.aa2.astype(int)
        self.bb1, self.aa1 = np.meshgrid(np.arange(0, self.m), np.arange(0, self.n), indexing='ij')
        self.bb2, self.aa2 = np.meshgrid(np.arange(self.m, self.n), np.arange(0, self.n), indexing='ij')

        self.t_switch = None
        self.nswitches = 0

    def eval_f(self, u, du, t):
        r"""
        Routine to evaluate the implicit representation of the problem, i.e., :math:`F(u, u', t)`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time t.
        du : dtype_u
            Current values of the derivative of the numerical solution at time t.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        f : dtype_f
            The right-hand side of f (contains two components).
        """

        dEqp, dSi1d, dEdp = du.diff[0 : self.m], du.diff[self.m : 2 * self.m], du.diff[2 * self.m : 3 * self.m]
        dSi2q, dDelta = du.diff[3 * self.m : 4 * self.m], du.diff[4 * self.m : 5 * self.m]
        dw, dEfd, dRF = (
            du.diff[5 * self.m : 6 * self.m],
            du.diff[6 * self.m : 7 * self.m],
            du.diff[7 * self.m : 8 * self.m],
        )
        dVR, dTM, dPSV = (
            du.diff[8 * self.m : 9 * self.m],
            du.diff[9 * self.m : 10 * self.m],
            du.diff[10 * self.m : 11 * self.m],
        )

        Eqp, Si1d, Edp = u.diff[0 : self.m], u.diff[self.m : 2 * self.m], u.diff[2 * self.m : 3 * self.m]
        Si2q, Delta = u.diff[3 * self.m : 4 * self.m], u.diff[4 * self.m : 5 * self.m]
        w, Efd, RF = u.diff[5 * self.m : 6 * self.m], u.diff[6 * self.m : 7 * self.m], u.diff[7 * self.m : 8 * self.m]
        VR, TM, PSV = (
            u.diff[8 * self.m : 9 * self.m],
            u.diff[9 * self.m : 10 * self.m],
            u.diff[10 * self.m : 11 * self.m],
        )

        Id, Iq = u.alg[0 : self.m], u.alg[self.m : 2 * self.m]
        V = u.alg[2 * self.m : 2 * self.m + self.n]
        TH = u.alg[2 * self.m + self.n : 2 * self.m + 2 * self.n]

        # Mechanical power input reduction event (replaces previous line outage event)
        # MATLAB reference: if t > tf1 && t < tf2 then reduce first machine mechanical input by 50%
        # Define event window parameters on first call
        if not hasattr(self, 'tf1'):
            self.tf1 = 0.1  # fault application
        if not hasattr(self, 'tf2'):
            self.tf2 = 0.9  # fault clearing

        # Select network (pre/post vs fault) according to MATLAB staging logic
        if self.tf1 < t < self.tf2:
            Ynet = self.YBus_fault
        else:
            Ynet = self.YBus

        # No mechanical power reduction event here; keep PC constant
        PC0 = self.PC.copy()

        # Update admittance polar components for current network
        self.Yang = np.angle(Ynet)
        self.Yabs = np.abs(Ynet)

        COI = np.sum(w * self.MH) / np.sum(self.MH)

        # Voltage-dependent active loads PL2, and voltage-dependent reactive loads QL2
        PL2 = np.array(self.PL)
        QL2 = np.array(self.QL)

        V = V.T

        # (partially) Vectorized calculations
        # Vectorized_angle1 = (
        #     np.array([TH.take(indices) for indices in self.bb1.T])
        #     - np.array([TH.take(indices) for indices in self.aa1.T])
        #     - self.Yang[: self.m, : self.n]
        # )
        # Vectorized_mag1 = (V[: self.m] * V[: self.n].reshape(-1, 1)).T * self.Yabs[: self.m, : self.n]

        # Fully vectorized calculations
        Vectorized_angle1 = TH[self.bb1] - TH[self.aa1] - self.Yang[:self.m, :self.n]
        Vectorized_mag1 = V[:self.m, np.newaxis] * V[np.newaxis, :self.n] * self.Yabs[:self.m, :self.n]


        sum1 = np.sum(Vectorized_mag1 * np.cos(Vectorized_angle1), axis=1)
        sum2 = np.sum(Vectorized_mag1 * np.sin(Vectorized_angle1), axis=1)

        VG = V[: self.m]
        THG = TH[: self.m]
        Angle_diff = Delta - THG

        # (partially) Vectorized calculations
        # Vectorized_angle2 = (
        #     np.array([TH.take(indices) for indices in self.bb2.T])
        #     - np.array([TH.take(indices) for indices in self.aa2.T])
        #     - self.Yang[self.m : self.n, : self.n]
        # )
        # Vectorized_mag2 = (V[self.m : self.n] * V[: self.n].reshape(-1, 1)).T * self.Yabs[self.m : self.n, : self.n]
       
        # Fully vectorized calculations
        Vectorized_angle2 = TH[self.bb2] - TH[self.aa2] - self.Yang[self.m:self.n, :self.n]
        Vectorized_mag2 = V[self.m:self.n, np.newaxis] * V[np.newaxis, :self.n] * self.Yabs[self.m:self.n, :self.n]

        sum3 = np.sum(Vectorized_mag2 * np.cos(Vectorized_angle2), axis=1)
        sum4 = np.sum(Vectorized_mag2 * np.sin(Vectorized_angle2), axis=1)

        # Initialise f
        f = self.dtype_f(self.init)

        t_switch = np.inf if self.t_switch is None else self.t_switch

        # Equations as list
        eqs = []
        eqs.append(
            (1.0 / self.Td0p)
            * (
                -Eqp
                - (self.Xd - self.Xdp)
                * (
                    Id
                    - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls) ** 2) * (Si1d + (self.Xdp - self.Xls) * Id - Eqp)
                )
                + Efd
            )
            - dEqp
        )  # (1)
        eqs.append((1.0 / self.Td0pp) * (-Si1d + Eqp - (self.Xdp - self.Xls) * Id) - dSi1d)  # (2)
        eqs.append(
            (1.0 / self.Tq0p)
            * (
                -Edp
                + (self.Xq - self.Xqp)
                * (
                    Iq
                    - ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls) ** 2) * (Si2q + (self.Xqp - self.Xls) * Iq + Edp)
                )
            )
            - dEdp
        )  # (3)
        eqs.append((1.0 / self.Tq0pp) * (-Si2q - Edp - (self.Xqp - self.Xls) * Iq) - dSi2q)  # (4)
        eqs.append(w - COI - dDelta)  # (5)
        eqs.append(
            (self.ws / (2.0 * self.H))
            * (
                TM
                - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp * Iq
                - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d * Iq
                - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp * Id
                + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q * Id
                - (self.Xqpp - self.Xdpp) * Id * Iq
                - self.Dm * (w - self.ws)
            )
            - dw
        )  # (6)
        eqs.append((1.0 / self.TE) * ((-(self.KE + self.Ax * np.exp(self.Bx * Efd))) * Efd + VR) - dEfd)  # (7)
        eqs.append((1.0 / self.TF) * (-RF + (self.KF / self.TF) * Efd) - dRF)  # (8)
        eqs.append(
            (1.0 / self.TA)
            * (-VR + self.KA * RF - ((self.KA * self.KF) / self.TF) * Efd + self.KA * (self.Vref - V[: self.m]))
            - dVR
        )  # (9)

        # Limitation of valve position Psv with limiter start
        if PSV[0] >= self.psv_max or t >= t_switch:
            # When limiter active, freeze first component dynamics (dPSV[0])
            _temp_dPSV_g1 = (1.0 / self.TSV[1]) * (
                -PSV[1] + PC0[1] - (1.0 / self.RD[1]) * (w[1] / self.ws - 1)
            ) - dPSV[1]
            _temp_dPSV_g2 = (1.0 / self.TSV[2]) * (
                -PSV[2] + PC0[2] - (1.0 / self.RD[2]) * (w[2] / self.ws - 1)
            ) - dPSV[2]
            eqs.append(np.array([dPSV[0], _temp_dPSV_g1, _temp_dPSV_g2]))
        else:
            eqs.append((1.0 / self.TSV) * (-PSV + PC0 - (1.0 / self.RD) * (w / self.ws - 1)) - dPSV)
        # Limitation of valve position Psv with limiter end

        eqs.append((1.0 / self.TCH) * (-TM + PSV) - dTM)  # (10)
        eqs.append(
            self.Rs * Id
            - self.Xqpp * Iq
            - ((self.Xqpp - self.Xls) / (self.Xqp - self.Xls)) * Edp
            + ((self.Xqp - self.Xqpp) / (self.Xqp - self.Xls)) * Si2q
            + VG * np.sin(Angle_diff)
        )  # (12)
        eqs.append(
            self.Rs * Iq
            + self.Xdpp * Id
            - ((self.Xdpp - self.Xls) / (self.Xdp - self.Xls)) * Eqp
            - ((self.Xdp - self.Xdpp) / (self.Xdp - self.Xls)) * Si1d
            + VG * np.cos(Angle_diff)
        )  # (13)
        eqs.append((Id * VG.T * np.sin(Angle_diff) + Iq * VG.T * np.cos(Angle_diff)) - PL2[0 : self.m] - sum1)  # (14)
        eqs.append((Id * VG.T * np.cos(Angle_diff) - Iq * VG.T * np.sin(Angle_diff)) - QL2[0 : self.m] - sum2)  # (15)
        eqs.append(-PL2[self.m : self.n] - sum3)  # (16)
        eqs.append(-QL2[self.m : self.n] - sum4)  # (17)
        eqs_flatten = [item for sublist in eqs for item in sublist]

        f.diff[: 11 * self.m] = eqs_flatten[0 : 11 * self.m]
        f.alg[: 2 * self.n + 2 * self.m] = eqs_flatten[11 * self.m :]
        return f

    def u_exact(self, t):
        r"""
        Returns the initial conditions at time :math:`t=0`.

        Parameters
        ----------
        t : float
            Time of the initial conditions.

        Returns
        -------
        me : dtype_u
            Initial conditions.
        """
        assert t == 0, 'ERROR: u_exact only valid for t=0'

        me = self.dtype_u(self.init)
        me.diff[0 : self.m] = self.Eqp0
        me.diff[self.m : 2 * self.m] = self.Si1d0
        me.diff[2 * self.m : 3 * self.m] = self.Edp0
        me.diff[3 * self.m : 4 * self.m] = self.Si2q0
        me.diff[4 * self.m : 5 * self.m] = self.D0
        me.diff[5 * self.m : 6 * self.m] = self.ws_vector
        me.diff[6 * self.m : 7 * self.m] = self.Efd0
        me.diff[7 * self.m : 8 * self.m] = self.RF0
        me.diff[8 * self.m : 9 * self.m] = self.VR0
        me.diff[9 * self.m : 10 * self.m] = self.TM0
        me.diff[10 * self.m : 11 * self.m] = self.PSV0
        me.alg[0 : self.m] = self.Id0
        me.alg[self.m : 2 * self.m] = self.Iq0
        me.alg[2 * self.m : 2 * self.m + self.n] = self.V0
        me.alg[2 * self.m + self.n : 2 * self.m + 2 * self.n] = self.TH0
        return me
    
    def _report_initial_residual(self):
        """
        Optional diagnostic: compute ||F(u0, du=0)||_2 to verify consistent initialization.
        """
        u0 = self.u_exact(0.0)
        # Build a zero derivative container (du): for index-1 DAE consistent IC, du not strictly needed;
        # still we set to zeros and inspect residual.
        du0 = self.dtype_u(self.init)
        f0 = self.eval_f(u0, du0, 0.0)
        diff_norm = np.linalg.norm(f0.diff)
        alg_norm = np.linalg.norm(f0.alg)
        total = (diff_norm**2 + alg_norm**2)**0.5
        print(f"[IC residual] ||F_diff||={diff_norm:.3e}  ||F_alg||={alg_norm:.3e}  ||F_total||={total:.3e}")

    def get_switching_info(self, u, t):
        r"""
        Provides information about the state function of the problem. When the state function changes its sign,
        typically an event occurs. So the check for an event should be done in the way that the state function
        is checked for a sign change. If this is the case, the intermediate value theorem states a root in this
        step.

        The state function for this problem is given by

        .. math::
           h(P_{SV,1}(t)) = P_{SV,1}(t) - P_{SV,1, max}

        for :math:`P_{SV,1,max}=1.0`.

        Parameters
        ----------
        u : dtype_u
            Current values of the numerical solution at time :math:`t`.
        t : float
            Current time of the numerical solution.

        Returns
        -------
        switch_detected : bool
            Indicates whether a discrete event is found or not.
        m_guess : int
            The index before the sign changes.
        state_function : list
            Defines the values of the state function at collocation nodes where it changes the sign.
        """

        switch_detected = False
        m_guess = -100
        for m in range(1, len(u)):
            h_prev_node = u[m - 1].diff[10 * self.m] - self.psv_max
            h_curr_node = u[m].diff[10 * self.m] - self.psv_max
            if h_prev_node < 0 and h_curr_node >= 0:
                switch_detected = True
                m_guess = m - 1
                break

        state_function = [u[m].diff[10 * self.m] - self.psv_max for m in range(len(u))]
        return switch_detected, m_guess, state_function

    def count_switches(self):
        """
        Setter to update the number of switches if one is found.
        """
        self.nswitches += 1
