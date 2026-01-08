# velocity_profile_gen.py
import numpy as np


def velocity_profile_gen(track, mu, vehicle_mass, Ux_max):
    """
    Python conversion of MATLAB: velocity_profile_gen.m

    Inputs:
        track: dict-like with:
            track["station"]   -> array s
            track["curvature"] -> array K (same length)

    Returns:
        Ux_final, Ux_steady, Ux_forward, lap_time
    """
    # Default parameters (kept EXACTLY from your MATLAB)
    g = 20.0  # NOTE: your MATLAB uses 20, not 9.81
    max_engine_force = vehicle_mass * g

    s = np.asarray(track["station"]).reshape(-1)
    K = np.asarray(track["curvature"]).reshape(-1)

    # Pass 1: Steady-state cornering
    Ux_steady = np.sqrt(mu * g / (np.abs(K) + 1e-6))
    Ux_steady = np.minimum(Ux_steady, Ux_max)

    # Pass 2: Forward integration (acceleration limited)
    Ux_forward = np.zeros_like(s, dtype=float)
    Ux_forward[0] = Ux_steady[0]

    for i in range(1, len(s)):
        ds = s[i] - s[i - 1]

        Fx_max = max_engine_force
        Fy_demand = vehicle_mass * (Ux_forward[i - 1] ** 2) * abs(K[i])
        Fx_available = max(0.0, Fx_max - 0.5 * Fy_demand)

        ax_available = Fx_available / vehicle_mass
        Ux_new = np.sqrt(Ux_forward[i - 1] ** 2 + 2.0 * ax_available * ds)

        Ux_forward[i] = min(Ux_new, Ux_steady[i], Ux_max)

    # Pass 3: Backward integration (braking limited)
    Ux_final = np.zeros_like(s, dtype=float)
    Ux_final[-1] = Ux_forward[-1]

    max_braking_force = mu * vehicle_mass * g

    for i in range(len(s) - 2, -1, -1):
        ds = s[i + 1] - s[i]

        Fy_demand = vehicle_mass * (Ux_final[i + 1] ** 2) * abs(K[i])
        Fx_brake = max_braking_force - 0.5 * Fy_demand
        ax_brake = -Fx_brake / vehicle_mass

        Ux_new = np.sqrt(Ux_final[i + 1] ** 2 - 2.0 * ax_brake * ds)
        Ux_final[i] = min(Ux_new, Ux_forward[i])

    # Estimate lap time
    ds = np.diff(s)
    v_avg = (Ux_final[:-1] + Ux_final[1:]) / 2.0
    dt = ds / v_avg
    lap_time = float(np.sum(dt))

    return Ux_final, Ux_steady, Ux_forward, lap_time
