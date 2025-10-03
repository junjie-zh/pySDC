# %%
import sys
sys.path.append("../../../../../../pySDC")
import pickle

from pathlib import Path
import numpy as np
import statistics
import pySDC.helpers.plot_helper as plt_helper
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
# from pySDC.implementations.controller_classes.controller_MPI import controller_MPI

# from pySDC.projects.DAE.problems.wscc9BusSystem import WSCC9BusSystem
from pySDC.projects.DAE.problems.wscc9BusSystem_lineOutage import WSCC9BusSystem
# from pySDC.projects.DAE.problems.ieee68_noSE import case68System

from pySDC.projects.DAE.sweepers.fullyImplicitDAE import FullyImplicitDAE

# from pySDC.projects.DAE.misc.HookClass_DAE import approx_solution_hook
# from pySDC.projects.DAE.misc.HookClass_DAE import error_hook
from pySDC.implementations.hooks.log_solution import LogSolution
from pySDC.implementations.hooks.log_work import LogWork

from pySDC.helpers.stats_helper import get_sorted
from pySDC.helpers.stats_helper import filter_stats

import time
import matplotlib.pyplot as plt

import scienceplots
plt.style.use(['science', 'no-latex'])


# --- change or reverse default color cycle ---
from cycler import cycler
import numpy as np


# cmap = plt.cm.plasma_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
# cmap = plt.cm.inferno_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
# cmap = plt.cm.magma_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
# cmap = plt.cm.viridis_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
cmap = plt.cm.twilight_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
# cmap = plt.cm.seismic_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
# cmap = plt.cm.cividis_r  # Choose colormap: plasma_r, inferno_r, magma_r, viridis_r, etc.
n_colors = 10  # Number of distinct colors in the cycle
colors = [cmap(i) for i in np.linspace(0, 1, n_colors)]
plt.rcParams['axes.prop_cycle'] = cycler(color=colors)


# Optional: Reverse the default color cycle
plt.rcParams['axes.prop_cycle'] = cycler(color=plt.rcParams['axes.prop_cycle'].by_key()['color'][::-1])

# %%


def single_run_wscc09(run_name="wscc09", time_step=1e-1, end_time=0.1, sweeper_type = FullyImplicitDAE, 
                                useMPI = False, restol=1e-12, newtontol=1e-10,
                                QI_type="IE", num_nodes=3):
    """
    A testing ground for the synchronous machine model
    """
    # Ensure data directory exists
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['dt'] = time_step

    # initialize sweeper parameters
    M_fix = num_nodes
    sweeper_params = {
        'num_nodes': M_fix,
        'quad_type': 'RADAU-RIGHT',
        # 'QI': 'IE',
        'QI': QI_type
    }

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = newtontol  # tollerance for implicit solver

    ## WSCC09
    m=3
    n=9
    Path("data").mkdir(parents=True, exist_ok=True)

    # problem_params['nvars'] = 11 * m + 2 * m + 2 * n

    # initialize step parameters
    step_params = dict()
    if(sweeper_type == FullyImplicitDAE):
        step_params['maxiter'] = 50
    else:
        step_params['maxiter'] = 1

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['hook_class'] = [LogSolution, LogWork]

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = WSCC9BusSystem
    description['problem_params'] = problem_params
    description['sweeper_class'] = sweeper_type
    description['sweeper_params'] = sweeper_params
    description['level_params'] = level_params
    description['step_params'] = step_params

    Path("data").mkdir(parents=True, exist_ok=True)

    # instantiate the controller
    
    controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)

    # set time parameters
    t0 = 0.0
    Tend = end_time

    # get initial values on finest level
    P = controller.MS[0].levels[0].prob
    uinit = P.u_exact(t0)

    # call main function to get things done...
    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

    # store results
    # sol = get_sorted(stats, type='approx_solution', sortby='time')
    # sol_dt = np.array([sol[i][0] for i in range(len(sol))])
    # sol_data = np.array([[sol[j][1][i] for j in range(len(sol))] for i in range(problem_params['nvars'])])
    niter = filter_stats(stats, type='niter')
    niter = np.fromiter(niter.values(), int)
    
    n_rhs = filter_stats(stats, type='work_rhs', sortby='time')
    n_newton = get_sorted(stats, type='work_newton', sortby='time')
    # n_newton = np.fromiter(n_newton.values(), int)

    t = np.array([me[0] for me in get_sorted(stats, type='u', sortby='time')])
    # print([me[1][11*m + 2*m:11*m + 2*m + n] for me in get_sorted(stats, type='approx_solution', sortby='time', recomputed=False)])
    sol = get_sorted(stats, type='u', sortby='time')
    sol_dt = np.array([sol[i][0] for i in range(len(sol))])
    # sol_data = np.array([[sol[j][1][i] for j in range(len(sol))] for i in range(P.nvars)])
    # V = np.array([ me[1].alg[2 * m + 6] for me in get_sorted(stats, type='u', sortby='time')])
    # TH = np.array([ me[1].alg[2 * m + n + 6] for me in get_sorted(stats, type='u', sortby='time')])
    # Extract voltage magnitudes for all buses
    V_all = np.array([me[1].alg[2 * m : 2 * m + n] for me in get_sorted(stats, type='u', sortby='time')])
    # Extract voltage angles for all buses  
    TH_all = np.array([me[1].alg[2 * m + n : 2 * m + 2 * n] for me in get_sorted(stats, type='u', sortby='time')])

    # V = np.array([me[1][11 * m + 2 * m : 11 * m + 2 * m + n] for me in get_sorted(stats, type='u', sortby='time')])[
    #     :, 6
    # ] # get Vbus6 mag

    # TH = np.array([me[1][11 * m + 2 * m  + n: 11 * m + 2 * m + 2 * n] for me in get_sorted(stats, type='u', sortby='time')])[
    #     :, 6
    # ] # get Vbus6 ang

    timing = get_sorted(stats, type='timing_run', sortby='time')

    print('Time to solution: %6.4f sec.' % timing[0][1])

    res = {
        "t": sol_dt,
        "V_all": V_all,            # all bus voltages (n_timesteps x n_buses)
        "TH_all": TH_all,          # all bus angles (n_timesteps x n_buses)
        "niter":niter,
        "n_newton":n_newton,
        "time_to_solution": timing[0][1],
        }
    return run_name, t, res 



# %%
import itertools
import pandas as pd
from tabulate import tabulate

# Assuming single_run is imported from the relevant module
# from my_module import single_run

def run_tests_and_save(case=""):
    # Ensure both data and output directories exist
    Path(f"data/{case}").mkdir(parents=True, exist_ok=True)
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    filename = f"data/{case}/{case}_iter_data.pkl"
    
    # Ensure the directory exists
    Path(f"data/{case}").mkdir(parents=True, exist_ok=True)
    
    # Define parameter ranges
    time_steps = [1e-3, 10e-3, 100e-3]
    # time_steps = [10e-3]
    end_time = 1.0  # changed end time
    # end_time = 0.2  # changed end time
    restols = [1e-3]
    # restols = [1e-3, 1e-6]
    # newtontols = [1e-8, 1-9]
    newtontols = [1e-6]
    QI_types = ["IE", "LU", "MIN-SR-FLEX"]
    # QI_types = ["MIN-SR-S"]
    # num_nodes_list = [3, 4, 5]
    # num_nodes_list = [5]
    num_nodes_list = [2, 3]

    # List to store test results
    data = []

    # Calculate total number of combinations for progress tracking
    total_combinations = len(list(itertools.product(
        time_steps, restols, newtontols, QI_types, num_nodes_list
    )))
    
    print(f"Starting test suite for case: {case}")
    print(f"Total combinations to run: {total_combinations}")
    print(f"Data will be saved to: {filename}")
    print("-" * 60)

    # Iterate over all parameter combinations
    current_run = 0
    for time_step, restol, newtontol, QI_type, num_nodes in itertools.product(
        time_steps, restols, newtontols, QI_types, num_nodes_list
    ):
        current_run += 1
        run_name = f"ts_{time_step}_nt_{newtontol}_QI_{QI_type}_nodes_{num_nodes}"
        
        print(f"[{current_run:2d}/{total_combinations}] Running: {run_name}")
        print(f"    Parameters: dt={time_step:.1e}, restol={restol:.1e}, "
              f"newtontol={newtontol:.1e}, QI={QI_type}, nodes={num_nodes}")
        
        start_time = time.time()
        
        if case == "case68":
            run_name, times, res = single_run_case68(
                run_name=run_name,
                time_step=time_step,
                end_time=end_time,
                restol=restol,
                newtontol=newtontol,
                QI_type=QI_type,
                num_nodes=num_nodes,
                sweeper_type=FullyImplicitDAE,
                useMPI=False)
        elif "wscc09" in case:
            run_name, times, res = single_run_wscc09(
                run_name=run_name,
                time_step=time_step,
                end_time=end_time,
                restol=restol,
                newtontol=newtontol,
                QI_type=QI_type,
                num_nodes=num_nodes,
                sweeper_type=FullyImplicitDAE,
                useMPI=False)
        else:
            raise RuntimeError("Case unsupported.")

        elapsed_time = time.time() - start_time
        
        # Extract full time series from niter and newton iterations
        niter_series = res["niter"]           # e.g., an array of iterations at each time
        newton_series = res["n_newton"]         # list of tuples (time, newton_iteration)

        # Compute total sums (over the full time series)
        total_niter = np.sum(niter_series)
        total_newton = sum(val for _, val in newton_series)

        print(f"    Completed in {elapsed_time:.2f}s - "
              f"Total iterations: {total_niter}, Total Newton: {total_newton}")
        print()

        # Save all information for later analysis
        data.append({
            "run_name": run_name,
            "time_step": time_step,
            "restol": restol,
            "newtontol": newtontol,
            "QI_type": QI_type,
            "num_nodes": num_nodes,
            "times": times,                # full time series (numpy array)
            "niter_series": niter_series,  # numpy array
            "newton_series": newton_series,  # list of (time, iteration)
            "total_niter": total_niter,
            "total_newton": total_newton,
            "voltages": res["V_all"],          # voltage data for all bus
            "angles": res["TH_all"],           # angle data for all bus
            "res": res                     # store complete results dictionary
        })

    # Save the complete data into a pickle file
    print(f"All {total_combinations} test cases completed!")
    print(f"Saving results to: {filename}")
    
    try:
        with open(filename, "wb") as f:
            pickle.dump(data, f)
        print(f"Test data successfully saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")
        raise
    
    print("=" * 60)
# %%
import math
import pickle
import pandas as pd
import numpy as np
from tabulate import tabulate

# import scienceplots
# plt.style.use(['science', 'ieee', 'no-latex'])

def load_data(filename="test_data.pkl"):
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data

def create_summary_table(data, t_threshold=0.0):
    summary_rows = []
    for record in data:
        time_step = record["time_step"]
        newtontol = record["newtontol"]
        QI_type = record["QI_type"]
        num_nodes = record["num_nodes"]
        total_niter = record["total_niter"]
        total_newton = record["total_newton"]

        # For niter: filter times >= t_threshold
        times = record["times"]
        niter_series = record["niter_series"]
        mask = times >= t_threshold
        niter_after = np.sum(niter_series[mask]) if np.any(mask) else 0

        # For newton iterations: filter based on the time component of each tuple
        newton_series = record["newton_series"]
        newton_after = sum(val for t, val in newton_series if t >= t_threshold)

        frac_niter_after = niter_after / total_niter if total_niter != 0 else np.nan
        frac_newton_after = newton_after / total_newton if total_newton != 0 else np.nan

        summary_rows.append({
            "time_step": time_step,
            "newtontol": newtontol,
            "QI_type": QI_type,
            "num_nodes": num_nodes,
            "total_niter": total_niter,
            f"niter_after_{t_threshold}": niter_after,
            "frac_niter_after": frac_niter_after,
            "total_newton": total_newton,
            f"newton_after_{t_threshold}": newton_after,
            "frac_newton_after": frac_newton_after
        })

    df_summary = pd.DataFrame(summary_rows)
    return df_summary

def plot_reference(ref_filename, dt, t_threshold, metric_label, ax):
    """
    Reads a reference CSV file (without timestamps), creates a time array 
    with step size dt, and plots data for t >= t_threshold.
    """
    ref_data = pd.read_csv(ref_filename, header=None, names=[metric_label])
    num_rows = len(ref_data)
    times = np.arange(0, (num_rows) * dt, dt)
    mask = times >= t_threshold
    ax.plot(times[mask], ref_data[metric_label].values[mask],
            'k--', linewidth=2, label=f"Reference TR ({dt*1e3:.0f}ms)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(metric_label)
    # ax.legend(fontsize='x-large')
    ax.grid(True)
    # ax.set_title(f"Reference {metric_label}")
    # ax.set_title(f"Reference {metric_label}")

def _save_figs(fig, basename, outdir="figs/wscc09LineOutage", dpi=300, close=True, **kwargs):
    """
    Save a matplotlib figure in both PDF and PNG formats.

    Args:
        fig: matplotlib.figure.Figure or None (uses plt.gcf()).
        basename: base filename without extension.
        outdir: output directory to save files (created if missing).
        dpi: resolution for PNG save.
        close: if True, close the figure after saving.
        **kwargs: passed to savefig.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if fig is None:
        import matplotlib.pyplot as _plt
        fig = _plt.gcf()

    pdf_path = outdir / f"{basename}.pdf"
    png_path = outdir / f"{basename}.png"
    # eps_path = outdir / f"{basename}.eps"
    
    fig.savefig(str(pdf_path), format="pdf", bbox_inches='tight', **kwargs)
    fig.savefig(str(png_path), format="png", dpi=dpi, bbox_inches='tight', **kwargs)
    # fig.savefig(str(eps_path), format="eps", bbox_inches='tight', **kwargs)
    if close:
        try:
            import matplotlib.pyplot as _plt
            _plt.close(fig)
        except Exception:
            pass


def plot_all_metrics(data, t_threshold=0.9):
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Group data by (QI_type, num_nodes)
    groups = {}
    for record in data:
        key = (record["QI_type"], record["num_nodes"])
        groups.setdefault(key, []).append(record)

    # Determine grid layout: total subplots = reference + number of groups.
    n_groups = len(groups)
    # total_plots = n_groups + 1
    total_plots = n_groups
    ncols = 3  # Fixed number of columns
    nrows = math.ceil(total_plots / ncols)

    # ---------- Plot for Newton Iterations ----------
    # Build grid as: rows = num_nodes, columns = QI types
    qis = sorted({k[0] for k in groups.keys()})
    nodes_list = sorted({k[1] for k in groups.keys()})

    fig_newton, axes_newton = plt.subplots(
        len(nodes_list), len(qis),
        figsize=(5 * len(qis), 4 * len(nodes_list)),
        squeeze=False
    )

    # Reference file (kept for consistency, not plotted here)
    ref_filename_newton = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_0.0001_newton_iterations.csv"
    dt_ref = 0.1e-3

    for r, num_nodes in enumerate(nodes_list):
        for c, QI_type in enumerate(qis):
            ax = axes_newton[r, c]
            records = groups.get((QI_type, num_nodes), [])
            for record in records:
                # label = f"ts={record['time_step']:.1e}, ntol={record['newtontol']:.1e}"
                label = f"dt={record['time_step']:.1e}"
                newton_series = record["newton_series"]
                times_plot = [t for t, _ in newton_series if t >= t_threshold]
                vals_plot = [val for t, val in newton_series if t >= t_threshold]
                ax.plot(times_plot, vals_plot, marker='o', linestyle='-', label=label)

            if r == 0:
                ax.set_title(f"QI={QI_type}")
            if c == 0:
                ax.set_ylabel(f"nfeval (nodes={num_nodes})")
            ax.set_xlabel("Time (s)")
            ax.legend()
            ax.grid(True, alpha=0.7)

            if not records:
                ax.set_visible(False)

    plt.tight_layout()
    fig_newton.subplots_adjust(top=0.93)
    _save_figs(fig_newton, "newton_iterations", dpi=100)

    # ---------- Plot for niter ----------
    fig_niter, axes_niter = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes_niter = axes_niter.flatten()

    # First subplot: reference for niter.
    # ref_filename_niter = "data/newton_iters_TR_1ms.csv"
    # plot_reference(ref_filename_niter, dt_ref, t_threshold, "niter", axes_niter[0])

    # Plot each configuration's niter.
    # for idx, ((QI_type, num_nodes), records) in enumerate(groups.items(), start=1):
    for idx, ((QI_type, num_nodes), records) in enumerate(groups.items(), start=0):
        ax = axes_niter[idx]
        for record in records:
            label = f"ts={record['time_step']:.1e}, ntol={record['newtontol']:.1e}"
            times_arr = record["times"]
            niter_arr = record["niter_series"]
            mask = times_arr >= t_threshold
            ax.plot(times_arr[mask], niter_arr[mask], marker='o', linestyle='-', label=label)
        ax.set_title(f"QI={QI_type}, nodes={num_nodes}")
        ax.set_title(f"niter: QI={QI_type}, nodes={num_nodes}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("iter")
        # ax.legend(fontsize='x-large')
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(total_plots, len(axes_niter)):
        axes_niter[j].set_visible(False)

    plt.tight_layout()
    fig_niter.subplots_adjust(top=0.93)
    # fig_niter.suptitle("niter (t  {:.1f} s)".format(t_threshold), y=1.02)
    fig_niter.suptitle("Iterations")
    # plt.show()
    _save_figs(fig_niter, "niter")


    plt.figure(figsize=(10, 6))
    ref_filename_niter = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_0.0001_newton_iterations.csv"
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    plot_reference(ref_filename_niter, dt_ref, t_threshold, "Newton iterations", ax)
    # plt.show()
    _save_figs(fig, "ref_TR_1ms_newton_iters_scipy")


def plot_newton_iterations(data, t_threshold=0.9, ref_filename="data/newton_iters_TR_1ms.csv"):
    # Read the reference solution from CSV.
    # Assume the CSV has one column (without header) with newton iteration counts.
    ref_data = pd.read_csv(ref_filename, header=None, names=["newton_iter"])
    # Create a time array for the reference solution (1ms time step)
    num_rows = len(ref_data)
    dt_ref = 1e-3
    ref_times = np.arange(dt_ref, (num_rows+1)*dt_ref, dt_ref)
    # Filter the reference for t >= t_threshold
    ref_mask = ref_times >= t_threshold
    ref_times_plot = ref_times[ref_mask]
    ref_newton = ref_data["newton_iter"].values[ref_mask]

    # For compact plotting, we group subplots by QI_type and num_nodes.
    # Identify unique groups.
    groups = {}
    for record in data:
        key = (record["QI_type"], record["num_nodes"])
        groups.setdefault(key, []).append(record)

    # Determine subplot grid size
    n_groups = len(groups)
    ncols = 2
    nrows = (n_groups + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 4*nrows), squeeze=False)
    axes = axes.flatten()

    for idx, ((QI_type, num_nodes), records) in enumerate(groups.items()):
        ax = axes[idx]
        # For each record in the group, plot newton iterations after t_threshold.
        for record in records:
            # Use label to indicate time_step and newtontol
            label = f"ts={record['time_step']:.1e}, ntol={record['newtontol']:.1e}"
            # Get the newton_series (list of tuples) and filter for t>=t_threshold
            newton_series = record["newton_series"]
            times_plot = [t for t, _ in newton_series if t >= t_threshold]
            newton_vals = [val for t, val in newton_series if t >= t_threshold]
            ax.plot(times_plot, newton_vals, marker='o', linestyle='-', label=label)

        # Plot the reference solution in each subplot
        ax.plot(ref_times_plot, ref_newton, 'k--', linewidth=2, label="Reference TR (1ms)")

        ax.set_title(f"QI_type = {QI_type}, num_nodes = {num_nodes}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Newton Iterations")
        ax.legend(fontsize='x-large')
        ax.grid(True)

    # Hide any unused subplots
    for j in range(idx+1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_sum_newton_iterations_by_timestep(data, t_threshold=0.0, 
                                           ref_filename="data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_0.0001_newton_iterations.csv", 
                                           dt_ref=1e-3):
    """
    Create a bar plot for each of three time steps (1e-3, 1e-2, 1e-1), 
    showing the sum of Newton iterations (for times >= t_threshold) for each configuration,
    and including the reference solution. The y-axis is shared (same ylim) and uses a log scale.
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Define the three time steps to show (sorted in ascending order)
    ts_values = [1e-3, 1e-2, 1e-1]
    
    # Load reference data from CSV and compute reference sum for times >= t_threshold.
    ref_data = pd.read_csv(ref_filename, header=None, names=["newton_iter"])
    num_rows = len(ref_data)
    # Create a time array for the reference solution with dt_ref spacing.
    ref_times = np.arange(0, num_rows * dt_ref, dt_ref)  # Start from 0, use num_rows steps
    ref_mask = ref_times >= t_threshold
    ref_sum = ref_data["newton_iter"].values[ref_mask].sum()
    
    # Prepare to compute global y-limits.
    global_sums = []
    # We'll store each subplot's config labels and sums in a dict keyed by time step.
    ts_results = {}
    
    # Loop over each specified time step and compute the sums.
    for ts in ts_values:
        records = [rec for rec in data if abs(rec["time_step"] - ts) < 1e-12]
        config_labels = []
        config_sums = []
        
        for record in records:
            newton_series = record["newton_series"]
            newton_after = sum(val for t, val in newton_series if t >= t_threshold)
            label = f"QI:{record['QI_type']}\nnodes:{record['num_nodes']}"
            # label = f"QI:{record['QI_type']}\nnodes:{record['num_nodes']}\nntol:{record['newtontol']:.1e}"
            config_labels.append(label)
            config_sums.append(newton_after)
        
        # Append the reference solution as an extra bar.
        config_labels.append("Reference\n(TR, 0.1ms)")
        config_sums.append(ref_sum)
        
        # Save this result for plotting.
        ts_results[ts] = (config_labels, config_sums)
        global_sums.extend(config_sums)
    
    # Determine global y-limits.
    # Ensure no zero values for log scale; if found, set a small positive minimum.
    positive_sums = [s for s in global_sums if s > 0]
    if not positive_sums:
        ymin = 1e-1
    else:
        ymin = min(positive_sums) * 0.8
    ymax = max(global_sums) * 1.2
    
    # Create subplots: 1 row with 3 columns.
    ncols = 3
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    
    # Plot each time step.
    for i, ts in enumerate(ts_values):
         ax = axes[i]
         config_labels, config_sums = ts_results[ts]
         indices = np.arange(len(config_labels))
         bars = ax.bar(indices, config_sums, color="lightblue", edgecolor='black', alpha=0.7)
         
         # Annotate bars.
         for bar in bars:
             height = bar.get_height()
             ax.annotate(f'{height:.1f}', 
                         xy=(bar.get_x() + bar.get_width()/2, height),
                         xytext=(0, 5), textcoords="offset points",
                         ha='center', va='bottom', fontsize=9)
         
         ax.set_xticks(indices)
         ax.set_xticklabels(config_labels, rotation=45, ha='center', fontsize=9)
         # set ylabel only for plots in the first column (they all share the same ylabel)
         if (i % ncols) == 0:
             ax.set_ylabel("Sum of nfeval")
         ax.set_title(f"Time Step = {ts:.0e}")
         ax.grid(True, which="both", ls="--", lw=0.5, alpha= 0.6)
         ax.set_yscale("log")
         ax.set_ylim(ymin, ymax)
    
    # fig.suptitle("Comparison of Sum of Total Newton Iterations \nfor Different Time Steps", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    # plt.show()
    # Save both PDF and PNG
    _save_figs(plt.gcf(), "newton_iterations_sum_comparison")
    # plt.close()


# # -- define test case
# test_case = "wscc09_lineOutage"
# # test_case = "case68"

# ## --- run test ---
# # run_tests_and_save(f"{test_case}")
# # Load saved test data
# data = load_data(f"data/{test_case}/{test_case}_iter_data.pkl")


# df_summary = create_summary_table(data, t_threshold=0.0)
# print("Summary Table:")
# print(tabulate(df_summary, headers="keys", tablefmt="psql", floatfmt=".3e"))
# df_summary.to_csv(f"data/{test_case}/{test_case}_iter_data_summary.csv", index=False)

# # # plot_newton_iterations(data, t_threshold=0.0, ref_filename="data/newton_iters_TR_1ms_scipy.csv")
# # # Plot Newton iterations and niter metrics
# # # plot_all_metrics(data, t_threshold=0.0)
# plot_sum_newton_iterations_by_timestep(data, t_threshold=0.00, ref_filename='data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_0.0001_V_.csv')


# plt.show()
def load_reference_voltage(filename, bus_idx=6):
    """
    Load reference voltage data from TR simulation CSV file.
    
    Args:
        filename: Path to CSV file with voltage data
        bus_idx: Bus index (0-based) to extract voltage for
    
    Returns:
        tuple: (times, voltages) arrays
    """
    data = pd.read_csv(filename)
    times = data.iloc[:, 0].values  # First column is time
    voltages = data.iloc[:, bus_idx].values  # Bus voltages start from column 1
    return times, voltages

def compute_rmse(ref_values, test_values, ref_times, test_times, mode='both'):
    """
    Compute RMSE between reference and test values with different interpolation strategies.
    
    Args:
        ref_values: Reference voltage values
        test_values: Test voltage values  
        ref_times: Reference time points
        test_times: Test time points
        mode: 'forward', 'backward', or 'both'
    
    Returns:
        dict or float: RMSE values
            - 'forward': Forward RMSE (reference interpolated to test grid)  
            - 'backward': Backward RMSE (test interpolated to reference grid)
            - If mode is single string, returns float
    """
    results = {}
    
    if mode in ['forward', 'both']:
        # Forward RMSE: interpolate reference to test time points
        if not np.array_equal(ref_times, test_times):
            ref_interp = np.interp(test_times, ref_times, ref_values)
        else:
            ref_interp = ref_values
        forward_rmse = np.sqrt(np.mean((ref_interp - test_values)**2))
        results['forward'] = forward_rmse
    
    if mode in ['backward', 'both']:
        # Backward RMSE: interpolate test to reference time points  
        if not np.array_equal(ref_times, test_times):
            test_interp = np.interp(ref_times, test_times, test_values)
        else:
            test_interp = test_values
        backward_rmse = np.sqrt(np.mean((ref_values - test_interp)**2))
        results['backward'] = backward_rmse
    
    if mode == 'both':
        return results
    else:
        return results[mode]

def plot_voltage_absolute_errors(data, bus_idx=5, t_start=0.08, t_end=0.2):
    """
    Plot absolute voltage errors between TR reference and SDC results.
    Similar structure to plot_voltage_comparison_with_rmse but shows |error|.
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Load reference data (finest time step as reference)
    ref_filename = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_5e-05_V_.csv"
    ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx+1)
    
    # Group SDC data by configuration
    groups = {}
    for record in data:
        key = (record["QI_type"], record["num_nodes"], record["time_step"])
        groups[key] = record
    
    # Create subplots for different time steps
    time_steps = sorted(set(record["time_step"] for record in data))
    n_ts = len(time_steps)
    
    fig, axes = plt.subplots(1, n_ts, figsize=(6 * n_ts, 6))
    if n_ts == 1:
        axes = [axes]
    
    for i, ts in enumerate(time_steps):
        ax = axes[i]
        
        # Plot SDC configurations for this time step
        ts_records = [(key, record) for key, record in groups.items() if key[2] == ts]
        
        for (QI_type, num_nodes, _), record in ts_records:
            # Extract voltage data from SDC results
            times_sdc = record["times"]
            if "voltages" in record:
                voltages_sdc = record["voltages"][:, bus_idx]  # Extract bus 6 from all voltages
            else:
                print(f"Warning: No voltage data found for {QI_type}, {num_nodes}, {ts}")
                continue
            
            # Interpolate SDC to reference time points for error calculation
            voltages_sdc_interp = np.interp(ref_times, times_sdc, voltages_sdc)
            
            # Compute absolute error
            abs_error = np.abs(voltages_sdc_interp - ref_voltages)
            
            # Compute RMSE for labeling
            rmse = compute_rmse(ref_voltages, voltages_sdc, ref_times, times_sdc)
            
            # Plot absolute error
            # label = f'QI:{QI_type}, nodes:{num_nodes}, ntol:{record["newtontol"]:.1e} (RMSE:{rmse:.2e})'
            label = f'QI:{QI_type}, nodes:{num_nodes}'
            ax.plot(ref_times, abs_error, '--', linewidth=2, 
                   marker='o', markersize=4, label=label)
        
           # Add horizontal reference line at error = 0.01
        ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
                alpha=0.7, label='Error = 0.01')

        # ax.set_title(f'Bus {bus_idx+1} Voltage Absolute Error (dt = {ts:.0e})')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Abs. Error (p.u.)')
        # ax.set_xlim(0.08, 0.52)
        ax.set_ylim(1e-9, 1)
        ax.set_yscale('log')  # Log scale for error visualization
        ax.legend(fontsize='x-large', frameon=True)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figs(fig, "voltage_absolute_errors", dpi=300)
    # plt.show()
    
    return fig
def plot_voltage_comparison_with_rmse(data, bus_idx=5, t_start=0.08, t_end=0.2):
    """
    Plot voltage comparison between TR reference and SDC results with forward/backward RMSE computation.
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Load reference data (finest time step as reference)
    ref_filename = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_5e-05_V_.csv"
    ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx+1)
    
    # Filter reference data for plotting range
    ref_mask = (ref_times >= t_start) & (ref_times <= t_end)
    ref_times_plot = ref_times[ref_mask]
    ref_voltages_plot = ref_voltages[ref_mask]
    
    # Group SDC data by configuration
    groups = {}
    for record in data:
        key = (record["QI_type"], record["num_nodes"], record["time_step"])
        groups[key] = record
    
    # Create subplots for different time steps
    time_steps = sorted(set(record["time_step"] for record in data))
    n_ts = len(time_steps)
    
    fig, axes = plt.subplots(1, n_ts, figsize=(6 * n_ts, 6))
    if n_ts == 1:
        axes = [axes]
    
    # Store RMSE results for summary table
    rmse_results = []
    
    for i, ts in enumerate(time_steps):
        ax = axes[i]
        
        # Plot reference
        ax.plot(ref_times_plot, ref_voltages_plot, 'k-', linewidth=3, 
                label='Reference (TR, dt=5e-05)', alpha=0.5)
        
        # Plot SDC configurations for this time step
        ts_records = [(key, record) for key, record in groups.items() if key[2] == ts]
        
        for (QI_type, num_nodes, _), record in ts_records:
            # Extract voltage data from SDC results
            times_sdc = record["times"]
            # Get voltage for bus 6 - need to extract from the results properly
            # Assuming V data is stored in record, but need to verify structure
            if "voltages" in record:
                voltages_sdc = record["voltages"][:, bus_idx]  # Extract bus 6 from all voltages
            else:
                print(f"Warning: No voltage data found for {QI_type}, {num_nodes}, {ts}")
                continue
            
            times_sdc_plot = times_sdc
            voltages_sdc_plot = voltages_sdc

            # Compute both forward and backward RMSE for full time range
            rmse_both = compute_rmse(ref_voltages, voltages_sdc, ref_times, times_sdc, mode='both')
            
            # Store RMSE results
            rmse_results.append({
                "time_step": ts,
                "QI_type": QI_type,
                "num_nodes": num_nodes,
                "newtontol": record["newtontol"],
                "forward_rmse": rmse_both['forward'],
                "backward_rmse": rmse_both['backward']
            })
            
            # Plot SDC result (show backward RMSE in label for consistency with original)
            label = f'QI:{QI_type}, nodes:{num_nodes}'
            # label = f'QI:{QI_type}, nodes:{num_nodes}, ntol:{record["newtontol"]:.1e} (RMSE:{rmse:.2e})'
            ax.plot(times_sdc_plot, voltages_sdc_plot, '--', linewidth=2, 
                   marker='o', markersize=4, label=label)
        
        ax.set_title(f'Bus {bus_idx+1} Voltage (dt = {ts:.0e}s)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage Magnitude (p.u.)')
        # ax.set_xlim(t_start, t_end)
        ax.set_xlim(0.08, 0.28)
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize='x-large', frameon=True)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figs(fig, "voltage_comparison_with_rmse", dpi=300)
    plt.show()
    
    # Create RMSE summary table
    rmse_df = pd.DataFrame(rmse_results)
    print("\nRMSE Summary Table:")
    print(tabulate(rmse_df, headers="keys", tablefmt="psql", floatfmt=".3e"))
    rmse_df.to_csv(f"data/{test_case}/voltage_rmse_summary.csv", index=False)
    
    return rmse_df

def plot_tr_reference_absolute_errors():
    """
    Plot absolute errors between different TR time steps and the finest reference.
    Similar to plot_tr_reference_comparison but shows |error| instead of voltages.
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Reference solution (finest time step)
    ref_dt = 5e-05
    ref_filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{ref_dt}_V_.csv"
    
    dt_values = [0.01, 0.001, 0.0001]
    bus_idx = 6
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Load reference data
    try:
        ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx)
    except FileNotFoundError:
        print(f"Error: Reference file not found: {ref_filename}")
        return
    
    # Store RMSE results
    rmse_results = []
    # Define color order - you can modify this list to reorder colors
    # color_order = [0, 2, 1]  # Example: use 1st, 3rd, 2nd colors from default cycle
    # color_order = [2, 1, 0]  # Alternative: reverse order
    color_order = [1, 0, 2]  # Alternative: different order

    for i, dt in enumerate(dt_values):
        filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
        try:
            times, voltages = load_reference_voltage(filename, bus_idx)
            
            # Interpolate test solution to reference time points
            voltages_interp = np.interp(ref_times, times, voltages)
            
            # Compute absolute error
            abs_error = np.abs(voltages_interp - ref_voltages)
            
            # Compute RMSE for labeling
            rmse = compute_rmse(ref_voltages, voltages, ref_times, times)
            rmse_results.append({
                "dt": dt,
                "dt_ms": dt * 1000,
                "rmse": rmse
            })
            
            # Plot absolute error
            # label = f'TR (dt = {dt*1000:.1f}ms, RMSE = {rmse:.2e})'
            label = f'TR (dt = {dt:.0e}s)'
            marker = 'o' if dt >= 0.001 else ''
            # Use color_order to select color from matplotlib's default cycle
            color_idx = color_order[i % len(color_order)]
            ax.plot(ref_times, abs_error, linewidth=2, label=label, 
                   marker=marker, markersize=4, color=f'C{color_idx}')

        except FileNotFoundError:
            print(f"Warning: Reference file not found: {filename}")
    
   # Add horizontal reference line at error = 0.01
    ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
               alpha=0.7, label='Error = 0.01')

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Abs. Error (p.u.)')
    # ax.set_title(f'Bus {bus_idx} Voltage Absolute Error vs Reference (dt = {ref_dt*1000:.2f}ms)')
    ax.set_yscale('log')  # Log scale for error visualization

    ax.legend()
    ax.grid(True, alpha=0.3)
    # ax.set_xlim(0.08, 0.52)
    ax.set_ylim(1e-9, 1)
    plt.tight_layout()
    _save_figs(fig, "tr_reference_absolute_errors", dpi=300)
    # plt.show()
    
    # # Print RMSE summary
    # rmse_df = pd.DataFrame(rmse_results)
    # print("\nTR Reference Absolute Error RMSE Summary:")
    # print(tabulate(rmse_df, headers="keys", tablefmt="psql", floatfmt=".3e"))
    
    # return rmse_df

def plot_tr_reference_comparison():
    """
    Plot comparison between different TR time steps to validate reference choice.
    Compute RMSE against the finest reference (dt=5e-05).
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Reference solution (finest time step)
    ref_dt = 5e-05
    ref_filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{ref_dt}_V_.csv"
    
    # dt_values = [0.0001, 0.001, 0.01]
    dt_values = [0.01, 0.001, 0.0001] # for plot appearance
    bus_idx = 6
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    # Load reference data
    try:
        ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx)
        ax.plot(ref_times, ref_voltages, 'k-', linewidth=3, 
               label=f'Reference TR (dt = {ref_dt:.0e}s)', alpha=0.8) # plot at the end
    except FileNotFoundError:
        print(f"Error: Reference file not found: {ref_filename}")
        return
    
    # Store RMSE results
    rmse_results = []
    
    # Define color order - you can modify this list to reorder colors
    # color_order = [0, 2, 1]  # Example: use 1st, 3rd, 2nd colors from default cycle
    # color_order = [2, 1, 0]  # Alternative: reverse order
    color_order = [1, 0, 2]  # Alternative: different order

    # Helper function to read newton iterations from CSV and compute total
    def load_newton_iterations_total(dt):
        """Load newton iterations data and compute total"""
        newton_filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_newton_iterations.csv"
        try:
            newton_data = pd.read_csv(newton_filename, header=None, names=["newton_iter"])
            total_newton = newton_data["newton_iter"].sum()
            return total_newton
        except FileNotFoundError:
            print(f"Warning: Newton iterations file not found: {newton_filename}")
            return None

    for i, dt in enumerate(dt_values):
        filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
        try:
            times, voltages = load_reference_voltage(filename, bus_idx)
            
            # Compute both forward and backward RMSE against reference
            rmse_both = compute_rmse(ref_voltages, voltages, ref_times, times, mode='both')
            
            # Load newton iterations total
            total_newton = load_newton_iterations_total(dt)
            
            rmse_results.append({
                "dt": dt,
                "dt_ms": dt * 1000,
                "forward_rmse": rmse_both['forward'],
                "backward_rmse": rmse_both['backward'],
                "total_newton": total_newton
            })
            
            # Plot with backward RMSE in label (test interpolated to reference grid)
            # label = f'TR (dt = {dt*1000:.1f}ms, RMSE = {rmse_both["backward"]:.2e})'
            label = f'TR (dt = {dt:.0e}s)'
            marker = 'o' if dt >= 0.001 else ''
            # Use color_order to select color from matplotlib's default cycle
            color_idx = color_order[i % len(color_order)]
            ax.plot(times, voltages, linewidth=2, label=label, marker=marker, 
                   color=f'C{color_idx}')
            
        except FileNotFoundError:
            print(f"Warning: Reference file not found: {filename}")
    
    # Add reference newton iterations total
    ref_total_newton = load_newton_iterations_total(ref_dt)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Voltage Magnitude (p.u.)')
    ax.set_title(f'Bus {bus_idx} Voltage')
    # ax.set_title(f'Bus {bus_idx} Voltage - TR Reference Comparison with RMSE')
    ax.legend(frameon=True)
    ax.grid(True, alpha=0.3)
    # ax.set_xlim(0.08, 0.2)
    # ax.set_xlim(0.08, 0.52)
    ax.set_xlim(0.08, 0.28)
    ax.set_ylim(0.85, 1.02)
    
    plt.tight_layout()
    _save_figs(fig, "tr_reference_comparison_with_rmse", dpi=300)
    plt.show()
    
    # Print RMSE summary
    rmse_df = pd.DataFrame(rmse_results)
    print("\nTR Reference Comparison RMSE Summary:")
    print(tabulate(rmse_df, headers="keys", tablefmt="psql", floatfmt=".6e"))
    
    # Print reference newton total for completeness
    if ref_total_newton is not None:
        print(f"\nReference TR (dt = {ref_dt:.0e}s) total_newton: {ref_total_newton}")
    
    # Save RMSE results
    rmse_df.to_csv(f"data/{test_case}/tr_reference_rmse_summary.csv", index=False)
    
    return rmse_df


def create_time_to_solution_summary(data):
    """
    Create a summary table showing time to solution for all SDC runs.
    
    Args:
        data: List of dictionaries containing test results
    
    Returns:
        pandas.DataFrame: Summary table with timing information
    """
    summary_rows = []
    
    for record in data:
        summary_rows.append({
            "run_name": record["run_name"],
            "time_step": record["time_step"],
            "QI_type": record["QI_type"],
            "num_nodes": record["num_nodes"],
            "newtontol": record["newtontol"],
            "restol": record["restol"],
            "time_to_solution": record["res"]["time_to_solution"],
            "total_niter": record["total_niter"],
            "total_newton": record["total_newton"]
        })
    
    df_timing = pd.DataFrame(summary_rows)
    
    # Sort by time step, then QI type, then num_nodes for better readability
    df_timing = df_timing.sort_values(['time_step', 'QI_type', 'num_nodes'])
    
    return df_timing

def plot_time_to_solution_comparison(data):
    """
    Create a bar plot comparing time to solution across different configurations.
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Filter data to exclude diverged cases (time > 800s) for plotting only
    filtered_data = [record for record in data 
                    if record["res"]["time_to_solution"] <= 800]
    
    if not filtered_data:
        print("Warning: No data points with time <= 800s found for plotting")
        return

    # TR reference data
    tr_reference_time = 73.8991  # seconds
    tr_reference_dt = 0.0001     # 0.1ms

    # Group by time step for separate subplots
    time_steps = sorted(set(record["time_step"] for record in filtered_data))
    n_ts = len(time_steps)
    
    fig, axes = plt.subplots(1, n_ts, figsize=(6 * n_ts, 6))
    if n_ts == 1:
        axes = [axes]
    
    for i, ts in enumerate(time_steps):
        ax = axes[i]
        
        # Get data for this time step
        ts_data = [record for record in filtered_data if record["time_step"] == ts]
        
        # Create configuration labels and times
        config_labels = []
        config_times = []
        
        # Add SDC configurations
        for record in ts_data:
            label = f"QI:{record['QI_type']}\nnodes:{record['num_nodes']}"
            config_labels.append(label)
            config_times.append(record["res"]["time_to_solution"])
        
        # Add TR reference
        config_labels.append("Reference\n(TR, 0.1ms)")
        config_times.append(tr_reference_time)
        
        # Create bar plot
        bars = ax.bar(range(len(config_labels)), config_times, 
                     color='lightblue', edgecolor='black', alpha=0.7)
        
        # Add value labels on bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            ax.annotate(f'{height:.2f}s',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xticks(range(len(config_labels)))
        ax.set_xticklabels(config_labels, rotation=45, ha='center', fontsize=9)
        ax.set_ylabel('Time to Solution (s)')
        ax.set_title(f'Time Step = {ts:.0e}')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figs(fig, "time_to_solution_comparison", dpi=300)


def plot_voltage_comparison_new(data, bus_idx=5, t_start=0.08, t_end=0.2):
    """
    Create 4 subplots showing voltage comparisons:
    - p1: SDC (QI:LU, nodes=2) vs Reference for all time
    - p2: Same as p1 but zoomed to [0.08, 0.2]
    - p3: TR different time steps vs Reference for all time
    - p4: Same as p3 but zoomed to [0.08, 0.2]
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Load reference data (finest time step as reference)
    ref_filename = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_5e-05_V_.csv"
    ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx+1)
    
    # TR time steps to compare
    dt_values = [0.01, 0.001, 0.0001]
    # color_order = [1, 0, 2]  # Color ordering
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # --- Plots 1 & 2: SDC (QI:LU, nodes=2) vs Reference ---
    for plot_idx, (ax, xlim) in enumerate([(axes[0], None), (axes[1], (0.08, 0.2))]):
        # Plot reference
        ax.plot(ref_times, ref_voltages, 'k-', linewidth=3, 
                label='Reference (TR, dt=5e-05s)', alpha=0.5)
        
        # Filter SDC data for QI:LU, nodes=2
        for record in data:
            if record["QI_type"] == "LU" and record["num_nodes"] == 2:
                ts = record["time_step"]
                times_sdc = record["res"]["t"]
                voltages_sdc = record["res"]["V_all"][:, bus_idx]
                
                label = f'SDC (QI:LU, nodes:2, dt={ts:.0e}s)'
                ax.plot(times_sdc, voltages_sdc, '--', linewidth=2, 
                       marker='o', markersize=4, label=label)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage Magnitude (p.u.)')
        ax.set_title(f'Bus {bus_idx+1} - SDC vs Reference' + 
                    ('' if xlim is None else ' (Zoomed)'))
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize='x-large', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # --- Plots 3 & 4: TR different time steps vs Reference ---
    for plot_idx, (ax, xlim) in enumerate([(axes[2], None), (axes[3], (0.08, 0.2))]):
        # Plot reference
        ax.plot(ref_times, ref_voltages, 'k-', linewidth=3, 
                label='Reference (TR, dt=5e-05s)', alpha=0.8)
        
        # Plot TR with different time steps
        for i, dt in enumerate(dt_values):
            filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
            try:
                times, voltages = load_reference_voltage(filename, bus_idx+1)
                label = f'TR (dt = {dt:.0e}s)'
                marker = 'o' if dt >= 0.001 else ''
                # color_idx = color_order[i % len(color_order)]
                ax.plot(times, voltages, linewidth=2, label=label, 
                       marker=marker, markersize=4)
            except FileNotFoundError:
                print(f"Warning: File not found: {filename}")
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage Magnitude (p.u.)')
        ax.set_title(f'Bus {bus_idx+1} - TR vs Reference' + 
                    ('' if xlim is None else ' (Zoomed)'))
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_ylim(0.85, 1.02)
        ax.legend(fontsize='x-large', frameon=True)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figs(fig, "voltage_comparison_new", dpi=300)
    plt.show()
    
    return fig


def plot_voltage_error_new(data, bus_idx=5):
    """
    Create 4 subplots showing absolute voltage errors:
    - p1: SDC (QI:LU, nodes=2) error vs Reference for all time
    - p2: Same as p1 but zoomed to [0.08, 0.2]
    - p3: TR different time steps error vs Reference for all time
    - p4: Same as p3 but zoomed to [0.08, 0.2]
    """
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Load reference data (finest time step as reference)
    ref_filename = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_5e-05_V_.csv"
    ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx+1)
    
    # TR time steps to compare
    dt_values = [0.01, 0.001, 0.0001]
    # color_order = [1, 0, 2]  # Color ordering
    # color_order = [0,1,2]  # Color ordering
    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']

    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    # --- Plots 1 & 2: SDC (QI:LU, nodes=2) error ---
    for plot_idx, (ax, xlim) in enumerate([(axes[0], None), (axes[1], (0.08, 0.2))]):
        # Filter SDC data for QI:LU, nodes=2
        for record in data:
            if record["QI_type"] == "LU" and record["num_nodes"] == 2:
                ts = record["time_step"]
                times_sdc = record["res"]["t"]
                voltages_sdc = record["res"]["V_all"][:, bus_idx]
                
                # Interpolate reference to SDC time points
                ref_voltages_interp = np.interp(times_sdc, ref_times, ref_voltages)
                abs_error = np.abs(voltages_sdc - ref_voltages_interp)
                
                label = f'SDC (QI:LU, nodes:2, dt={ts:.0e}s)'
                ax.plot(times_sdc, abs_error, '--', linewidth=2, 
                       marker='o', markersize=4, label=label)
        
        # Add horizontal reference line at error = 0.01
        ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Error = 0.01')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Abs. Error (p.u.)')
        ax.set_title(f'Bus {bus_idx+1} - SDC Error' + 
                    ('' if xlim is None else ' (Zoomed)'))
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_yscale('log')
        ax.set_ylim(1e-9, 1)
        ax.legend(fontsize='x-large', frameon=True)
        ax.grid(True, alpha=0.3)
    
    # --- Plots 3 & 4: TR different time steps error ---
    for plot_idx, (ax, xlim) in enumerate([(axes[2], None), (axes[3], (0.08, 0.2))]):
        # Plot TR with different time steps
        for i, dt in enumerate(dt_values):
            filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
            try:
                times, voltages = load_reference_voltage(filename, bus_idx+1)
                
                # Interpolate to reference time points
                voltages_interp = np.interp(ref_times, times, voltages)
                abs_error = np.abs(voltages_interp - ref_voltages)
                
                label = f'TR (dt = {dt:.0e}s)'
                marker = 'o' if dt >= 0.001 else ''
                # color_idx = color_order[i % len(color_order)]
                ax.plot(ref_times, abs_error, linewidth=2, label=label, 
                    marker=marker, markersize=4)                
            except FileNotFoundError:
                print(f"Warning: File not found: {filename}")
        
        # Add horizontal reference line at error = 0.01
        ax.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
                   alpha=0.7, label='Error = 0.01')
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Abs. Error (p.u.)')
        ax.set_title(f'Bus {bus_idx+1} - TR Error' + 
                    ('' if xlim is None else ' (Zoomed)'))
        if xlim is not None:
            ax.set_xlim(xlim)
        ax.set_yscale('log')
        ax.set_ylim(1e-9, 1)
        ax.legend(fontsize='x-large', frameon=True)
        # ax.legend(fontsize='x-large', frameon=True)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    _save_figs(fig, "voltage_error_new", dpi=300)
    plt.show()
    
    return fig
def plot_voltage_comparison_with_inset(data, bus_idx=5, zoom_xlim=(0.08, 0.2)):
    """
    Create 2 subplots with insets showing voltage comparisons:
    - Left: SDC (QI:LU, nodes=2) vs Reference with zoomed inset
    - Right: TR different time steps vs Reference with zoomed inset
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Load reference data (finest time step as reference)
    ref_filename = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_5e-05_V_.csv"
    ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx+1)
    
    # TR time steps to compare
    dt_values = [0.01, 0.001, 0.0001]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(7.16*2, 3.5*1.5))
    
    # --- Left plot: SDC vs Reference ---
    ax_left = axes[0]
    
    # Plot reference
    ax_left.plot(ref_times, ref_voltages, 'k-', linewidth=3, 
                 label='Reference (TR, dt=5e-05s)', alpha=0.5)
    
    # Filter SDC data for QI:LU, nodes=2
    for record in data:
        if record["QI_type"] == "LU" and record["num_nodes"] == 2:
            ts = record["time_step"]
            times_sdc = record["res"]["t"]
            voltages_sdc = record["res"]["V_all"][:, bus_idx]
            
            label = f'SDC (QI:LU, nodes:2, dt={ts:.0e}s)'
            ax_left.plot(times_sdc, voltages_sdc, '--', linewidth=2, 
                        marker='o', markersize=4, label=label)
    
    ax_left.set_xlabel('Time (s)')
    ax_left.set_ylabel('Voltage Magnitude (p.u.)')
    # ax_left.set_title(f'Bus {bus_idx+1} - SDC vs Reference')
    ax_left.set_title(f'Bus {bus_idx+1} Voltage - SDC')
    ax_left.set_ylim(0.85, 1.05)
    ax_left.legend(fontsize='large', frameon=True, loc='lower right')
    ax_left.grid(True, alpha=0.3)
    
    # Create inset for left plot
    axins_left = inset_axes(ax_left, width="40%", height="40%", loc='upper center')
    axins_left.plot(ref_times, ref_voltages, 'k-', linewidth=2, alpha=0.5)
    for record in data:
        if record["QI_type"] == "LU" and record["num_nodes"] == 2:
            times_sdc = record["res"]["t"]
            voltages_sdc = record["res"]["V_all"][:, bus_idx]
            axins_left.plot(times_sdc, voltages_sdc, '--', linewidth=2, 
                           marker='o', markersize=3)
    
    axins_left.set_xlim(zoom_xlim)
    axins_left.set_ylim(0.85, 1.02)
    axins_left.grid(True, alpha=0.3)
    mark_inset(ax_left, axins_left, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
    
    # --- Right plot: TR vs Reference ---
    ax_right = axes[1]
    
    # Plot reference
    ax_right.plot(ref_times, ref_voltages, 'k-', linewidth=3, 
                  label='Reference (TR, dt=5e-05s)', alpha=0.5)
    
    # Plot TR with different time steps
    for i, dt in enumerate(dt_values):
        filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
        try:
            times, voltages = load_reference_voltage(filename, bus_idx+1)
            label = f'TR (dt = {dt:.0e}s)'
            marker = 'o' if dt >= 0.001 else ''
            ax_right.plot(times, voltages, linewidth=2, label=label, 
                         marker=marker, markersize=4)
        except FileNotFoundError:
            print(f"Warning: File not found: {filename}")
    
    ax_right.set_xlabel('Time (s)')
    ax_right.set_ylabel('Voltage Magnitude (p.u.)')
    # ax_right.set_title(f'Bus {bus_idx+1} - TR vs Reference')
    ax_right.set_title(f'Bus {bus_idx+1} Voltage - TR')
    ax_right.set_ylim(0.85, 1.05)
    ax_right.legend(fontsize='large', frameon=True, loc='lower right')
    ax_right.grid(True, alpha=0.3)
    
    # Create inset for right plot
    axins_right = inset_axes(ax_right, width="40%", height="40%", loc='upper center')
    axins_right.plot(ref_times, ref_voltages, 'k-', linewidth=2, alpha=0.8)
    for i, dt in enumerate(dt_values):
        filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
        try:
            times, voltages = load_reference_voltage(filename, bus_idx+1)
            marker = 'o' if dt >= 0.001 else ''
            axins_right.plot(times, voltages, linewidth=2, marker=marker, markersize=3)
        except FileNotFoundError:
            pass
    
    axins_right.set_xlim(zoom_xlim)
    axins_right.set_ylim(0.85, 1.02)
    axins_right.grid(True, alpha=0.3)
    mark_inset(ax_right, axins_right, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
    
    plt.tight_layout()
    _save_figs(fig, "voltage_comparison_with_inset", dpi=300)
    plt.show()
    
    return fig


def plot_voltage_error_with_inset(data, bus_idx=5, zoom_xlim=(0.08, 0.2)):
    """
    Create 2 subplots with insets showing absolute voltage errors:
    - Left: SDC (QI:LU, nodes=2) error vs Reference with zoomed inset
    - Right: TR different time steps error vs Reference with zoomed inset
    """
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    
    # Create output directory if it doesn't exist
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Load reference data (finest time step as reference)
    ref_filename = "data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_5e-05_V_.csv"
    ref_times, ref_voltages = load_reference_voltage(ref_filename, bus_idx+1)
    
    # TR time steps to compare
    dt_values = [0.01, 0.001, 0.0001]
    
    # Create figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(7.16*2, 3.5*1.5))
    
    # --- Left plot: SDC error ---
    ax_left = axes[0]
    
    # Filter SDC data for QI:LU, nodes=2
    for record in data:
        if record["QI_type"] == "LU" and record["num_nodes"] == 2:
            ts = record["time_step"]
            times_sdc = record["res"]["t"]
            voltages_sdc = record["res"]["V_all"][:, bus_idx]
            
            # Interpolate reference to SDC time points
            ref_voltages_interp = np.interp(times_sdc, ref_times, ref_voltages)
            abs_error = np.abs(voltages_sdc - ref_voltages_interp)
            
            label = f'SDC (QI:LU, nodes:2, dt={ts:.0e}s)'
            ax_left.plot(times_sdc, abs_error, '--', linewidth=2, 
                        marker='o', markersize=4, label=label)
    
    # Add horizontal reference line at error = 0.01
    ax_left.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
                    alpha=0.7, label='Error = 0.01')
    
    ax_left.set_xlabel('Time (s)')
    ax_left.set_ylabel('Abs. Error (p.u.)')
    ax_left.set_title(f'Bus {bus_idx+1} Voltage - SDC Error')
    ax_left.set_yscale('log')
    ax_left.set_ylim(1e-9, 1e5)
    ax_left.legend(fontsize='large', frameon=True, loc='lower right')
    ax_left.grid(True, alpha=0.3)
    
    # Create inset for left plot
    axins_left = inset_axes(ax_left, width="40%", height="40%", loc='upper center')
    for record in data:
        if record["QI_type"] == "LU" and record["num_nodes"] == 2:
            times_sdc = record["res"]["t"]
            voltages_sdc = record["res"]["V_all"][:, bus_idx]
            ref_voltages_interp = np.interp(times_sdc, ref_times, ref_voltages)
            abs_error = np.abs(voltages_sdc - ref_voltages_interp)
            axins_left.plot(times_sdc, abs_error, '--', linewidth=2, 
                           marker='o', markersize=3)
    
    axins_left.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axins_left.set_xlim(zoom_xlim)
    axins_left.set_yscale('log')
    axins_left.set_ylim(1e-9, 1)
    axins_left.grid(True, alpha=0.3)
    mark_inset(ax_left, axins_left, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
    
    # --- Right plot: TR error ---
    ax_right = axes[1]
    
    # Plot TR with different time steps
    for i, dt in enumerate(dt_values):
        filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
        try:
            times, voltages = load_reference_voltage(filename, bus_idx+1)
            
            # Interpolate to reference time points
            voltages_interp = np.interp(ref_times, times, voltages)
            abs_error = np.abs(voltages_interp - ref_voltages)
            
            label = f'TR (dt = {dt:.0e}s)'
            marker = 'o' if dt >= 0.001 else ''
            ax_right.plot(ref_times, abs_error, linewidth=2, label=label, 
                         marker=marker, markersize=4)
        except FileNotFoundError:
            print(f"Warning: File not found: {filename}")
    
    # Add horizontal reference line at error = 0.01
    ax_right.axhline(y=0.01, color='red', linestyle='--', linewidth=2, 
                     alpha=0.7, label='Error = 0.01')
    
    ax_right.set_xlabel('Time (s)')
    ax_right.set_ylabel('Abs. Error (p.u.)')
    ax_right.set_title(f'Bus {bus_idx+1} Voltage - TR Error')
    ax_right.set_yscale('log')
    ax_right.set_ylim(1e-9, 1e5)
    ax_right.legend(fontsize='large', frameon=True, loc='lower right')
    ax_right.grid(True, alpha=0.3)
    
    # Create inset for right plot
    axins_right = inset_axes(ax_right, width="40%", height="40%", loc='upper center')
    for i, dt in enumerate(dt_values):
        filename = f"data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_{dt}_V_.csv"
        try:
            times, voltages = load_reference_voltage(filename, bus_idx+1)
            voltages_interp = np.interp(ref_times, times, voltages)
            abs_error = np.abs(voltages_interp - ref_voltages)
            marker = 'o' if dt >= 0.001 else ''
            axins_right.plot(ref_times, abs_error, linewidth=2, 
                            marker=marker, markersize=3)
        except FileNotFoundError:
            pass
    
    axins_right.axhline(y=0.01, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    axins_right.set_xlim(zoom_xlim)
    axins_right.set_yscale('log')
    axins_right.set_ylim(1e-9, 1)
    axins_right.grid(True, alpha=0.3)
    mark_inset(ax_right, axins_right, loc1=2, loc2=4, fc="none", ec="0.5", linestyle='--')
    
    plt.tight_layout()
    _save_figs(fig, "voltage_error_with_inset", dpi=300)
    plt.show()
    
    return fig
# Update the main execution section:
if __name__ == "__main__":
    # Create figs directory
    Path("figs/wscc09LineOutage").mkdir(parents=True, exist_ok=True)
    
    # Define test case
    test_case = "wscc09_lineOutage"
    
    # Run tests if data doesn't exist
    # data_file = f"data/{test_case}/{test_case}_iter_data_20250923.pkl"
    data_file = f"data/{test_case}/{test_case}_iter_data.pkl"
    if not Path(data_file).exists():
        print("Running test suite...")
        run_tests_and_save(f"{test_case}")
    
    # Load saved test data
    data = load_data(data_file)
    
    # Generate summary table
    df_summary = create_summary_table(data, t_threshold=0.0)
    print("Summary Table:")
    print(tabulate(df_summary, headers="keys", tablefmt="psql", floatfmt=".3e"))
    df_summary.to_csv(f"data/{test_case}/{test_case}_iter_data_summary.csv", index=False)
    
    # Generate time to solution summary
    df_timing = create_time_to_solution_summary(data)
    print("\nTime to Solution Summary:")
    print(tabulate(df_timing, headers="keys", tablefmt="psql", floatfmt=".3f"))
    df_timing.to_csv(f"data/{test_case}/{test_case}_timing_summary.csv", index=False)
    
    # Plot time to solution comparison
    plot_time_to_solution_comparison(data)

    # Plot TR reference comparison
    plot_tr_reference_comparison()
    plot_tr_reference_absolute_errors()
    # Plot voltage comparison with RMSE
    # rmse_df = plot_voltage_comparison_with_rmse(data, bus_idx=5, t_start=0.08, t_end=0.52)
    # Plot voltage absolute errors
    # plot_voltage_absolute_errors(data, bus_idx=5, t_start=0.08, t_end=0.52)
    # plot_voltage_comparison_new(data, bus_idx=5)
    # plot_voltage_error_new(data, bus_idx=5)
    plot_voltage_comparison_with_inset(data, bus_idx=5, zoom_xlim=(0.08, 0.2))
    plot_voltage_error_with_inset(data, bus_idx=5, zoom_xlim=(0.08, 0.2))

    # Plot iteration metrics
    plot_all_metrics(data, t_threshold=0.0)
    plot_sum_newton_iterations_by_timestep(data, t_threshold=0.0, ref_filename="data/wscc09_lineOutage/event_IEEE9Bus_lineOutage_tF_0_1_0_9_dt_0.0001_newton_iterations.csv")
    
    print(f"\nAll plots saved to figs/wscc09LineOutage/")
    print(f"Data summaries saved to data/{test_case}/")
