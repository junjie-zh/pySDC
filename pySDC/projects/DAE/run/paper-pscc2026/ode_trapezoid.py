import time
import numpy as np
from scipy.optimize import root


def ode_trapezoid(odefun, tspan, x0, mass_diag=None, max_step=None,
                  progress=True, progress_interval=0.5,
                  newton_tol=1e-6, newton_max=1000):
    """
    Implicit trapezoidal solver for semi-explicit DAEs with diagonal mass matrix.
    Solves: M*(x_{n+1}-x_n) = h/2*( f(t_n,x_n) + f(t_{n+1},x_{n+1}) )

    Parameters
    ----------
    odefun : callable
        f(t, x) returning residuals/derivatives matching mass layout.
    tspan : (t0, tf) or 1D array
        If tuple/list of length 2, you must provide max_step; uniform steps used.
        If array, it is the explicit time grid.
    x0 : 1D ndarray
        Initial state.
    mass_diag : 1D ndarray or None
        Diagonal of mass matrix M (len = len(x0)). If None, identity is used.
    max_step : float or None
        Fixed step size when tspan is (t0, tf).
    progress : bool
        Print progress.
    progress_interval : float
        Seconds between progress prints.
    newton_tol : float
        Tolerance for the root solver (xtol – MINPACK).
    newton_max : int
        Max function evaluations per step.

    Returns
    -------
    t : (nt,) ndarray
        Time grid.
    X : (nt, nx) ndarray
        Solution trajectory, rows correspond to t.
    iter_t : (nt-1,) ndarray
        Time points t[1:] associated with root solves.
    iter_counts : (nt-1,) ndarray
        Function evaluation counts per step (nfev from MINPACK 'hybr').
    """
    # Build time grid
    if isinstance(tspan, (tuple, list)) and len(tspan) == 2:
        if max_step is None or max_step <= 0:
            raise ValueError("max_step required and must be > 0 when tspan = (t0, tf).")
        t0, tf = float(tspan[0]), float(tspan[1])
        nsteps = int(np.floor((tf - t0) / max_step + 1e-12))
        t = t0 + np.arange(nsteps + 1) * max_step
        if t[-1] < tf - 1e-14:
            t = np.concatenate([t, [tf]])
    else:
        t = np.asarray(tspan, dtype=float).ravel()
        if np.any(np.diff(t) <= 0):
            raise ValueError("tspan must be strictly increasing.")

    nt = len(t)
    nx = len(x0)

    if mass_diag is None:
        mass_diag = np.ones(nx, dtype=float)
    else:
        mass_diag = np.asarray(mass_diag, dtype=float)
        if mass_diag.shape != (nx,):
            raise ValueError("mass_diag must be a 1D array with length len(x0).")

    X = np.zeros((nt, nx), dtype=float)
    X[0, :] = x0.astype(float)

    iter_counts = np.zeros(nt - 1, dtype=int)

    if progress:
        t_start = time.time()
        last_print = 0.0
        print(f"Trapezoidal: {nt-1} steps")

    f_prev = odefun(t[0], X[0, :])

    # Mask to avoid dividing by zero on algebraic rows
    diff_mask = (mass_diag != 0.0)

    for k in range(1, nt):
        h = t[k] - t[k - 1]
        x_prev = X[k - 1, :].copy()
        f_current_prev = f_prev

        # Predictor: explicit Euler on differential rows only
        dx_pred = np.zeros(nx, dtype=float)
        dx_pred[diff_mask] = f_current_prev[diff_mask] / mass_diag[diff_mask]
        x_guess = x_prev + h * dx_pred

        # Residual for trapezoid
        def residual(xn):
            fn = odefun(t[k], xn)
            return mass_diag * (xn - x_prev) - 0.5 * h * (f_current_prev + fn)
        

        # Check if x_guess already satisfies tolerance
        res_guess = residual(x_guess)
        if np.linalg.norm(res_guess) < newton_tol:
            # Use x_guess directly, no need for root solve
            X[k, :] = x_guess
            iter_counts[k - 1] = 0  # Count the residual evaluation
        else:
            # Root solve
            sol = root(residual, x_guess, method='hybr', tol=newton_tol, options=dict(maxfev=newton_max))
            # sol = root(residual, x0, method='hybr', tol=newton_tol, options=dict(maxfev=newton_max))
            if not sol.success:
                raise RuntimeError(f"Root solve failed at step {k} (t={t[k]:.6g}): {sol.message}")
            
            X[k, :] = sol.x
            iter_counts[k - 1] = sol.nfev

        # Prepare for next step
        f_prev = odefun(t[k], X[k, :])

        # Progress print
        if progress:
            now = time.time()
            if (now - t_start - last_print) >= progress_interval or k == nt - 1:
                frac = k / (nt - 1)
                elapsed = now - t_start
                eta = elapsed / max(frac, 1e-12) - elapsed
                print(f"\rProgress: {100.0*frac:5.1f}% Step {k}/{nt-1} "
                      f"Elapsed: {elapsed:6.1f}s ETA: {eta:6.1f}s nfev: {iter_counts[k - 1]}", end="")
                last_print = elapsed
                if k == nt - 1:
                    print()

    iter_t = t[1:]
    return t, X, iter_t, iter_counts