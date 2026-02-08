import os
import numpy as np
from scipy.interpolate import splprep, splev
from scipy.signal import savgol_filter
from pal.utilities.math import wrap_to_pi

class Path2D:
    """
    Waypoints as (N,2). Builds arc-length s_i and heading psi_r(s_i), curvature kappa(s_i).
    """
    def __init__(self, waypoints_xy: np.ndarray, Ts):

        assert waypoints_xy.ndim == 2 and waypoints_xy.shape[1] == 2

        self.wp = waypoints_xy
        self.N = self.wp.shape[0]
        self.last_i = 0
        self.last_s = 0.0

        # Compute arc length parameter s
        dx = np.diff(self.wp[:, 0])
        dy = np.diff(self.wp[:, 1])
        ds = np.hypot(dx, dy)
        self.s = np.hstack(([0.0], np.cumsum(ds)))

        # Normalize parameter u ∈ [0,1]
        u = self.s / (self.s[-1] + 1e-9)

        # ---- Fit smooth spline through waypoints ----
        # s controls smoothing: 0 = exact fit, >0 = smoother
        tck, _ = splprep([self.wp[:, 0], self.wp[:, 1]], u=u, s=0.3)

        # Sample uniformly along arc length
        s_uniform = np.linspace(0, self.s[-1], self.N)
        u_uniform = s_uniform / (self.s[-1] + 1e-9)

        # Evaluate spline position & derivatives
        x, y = splev(u_uniform, tck, der=0)
        dx, dy = splev(u_uniform, tck, der=1)
        ddx, ddy = splev(u_uniform, tck, der=2)

        # Store smoothed centerline
        self.x = np.array(x)
        self.y = np.array(y)
        self.s = s_uniform

        # Heading psi
        self.psi = np.arctan2(dy, dx)

        # Curvature kappa
        denom = (dx * dx + dy * dy) ** 1.5 + 1e-9
        self.kappa = (dx * ddy - dy * ddx) / denom

        # ---- Extra smoothing on curvature to prevent spikes ----
        win = max(9, (self.N // 20) | 1)  # adaptive odd window size
        self.kappa = savgol_filter(self.kappa, window_length=win, polyorder=3)

        # ---- OPTIONAL: limit curvature rate of change ----
        kappa_lim = np.zeros_like(self.kappa)
        rate_max = 3.0  # max curvature change rate (tune if needed)

        for i in range(1, len(self.kappa)):
            dk = np.clip(self.kappa[i] - kappa_lim[i - 1], -rate_max * Ts, rate_max * Ts)
            kappa_lim[i] = kappa_lim[i - 1] + dk

        self.kappa = kappa_lim

        # ---- Save debug CSV ----
        np.savetxt("path2d_psi.csv", self.psi, delimiter=",", header="psi", comments="")
        np.savetxt("path2d_kappa.csv", self.kappa, delimiter=",", header="kappa", comments="")

        # Optional speed profile (computed offline on demand)
        self.v_ref = None


    def _closest_segment(self, p, i_center=None, window=50):
        # search only near the last best segment
        if i_center is None:
            i_center = self.last_i

        i0 = max(0, i_center - window)
        i1 = min(self.N - 2, i_center + window)

        best = (1e9, i_center, 0.0, None)
        for i in range(i0, i1 + 1):
            p0 = self.wp[i]
            p1 = self.wp[i+1]
            v = p1 - p0
            L2 = float(v @ v)
            if L2 < 1e-12:
                continue
            t = float(np.clip(((p - p0) @ v) / L2, 0.0, 1.0))
            proj = p0 + t * v
            d = float(np.linalg.norm(p - proj))
            if d < best[0]:
                best = (d, i, t, proj)
        return best

    def project_frenet(self, x, y, psi_vehicle, Ts, v_max):
        """
        Returns Frenet (s, ey, epsi, psi_r, kappa) at the closest projection.
        """
        p = np.array([x, y], dtype=float)
        dist, i, t, proj = self._closest_segment(p, i_center=self.last_i, window=100)
        self.last_i = i

        # s at projection
        seg_len = self.s[i+1] - self.s[i]
        s_proj = self.s[i] + t * seg_len
        # reference heading/curvature at nearest waypoint index
        # (simple: use i; better: interpolate by s_proj)
        psi_r = self.psi[i]
        kappa = self.kappa[i]

        # normal vector
        n = np.array([-np.sin(psi_r), np.cos(psi_r)])
        ey = float(n @ (p - proj))

        epsi = wrap_to_pi(psi_vehicle - psi_r)
        # ds_max = 2.0 * (v_max + 0.1) * Ts  # tune

        # s_raw = s_proj
        # s_proj = np.clip(s_raw, self.last_s - ds_max, self.last_s + ds_max)

        # enforce no going backward (recommended for racing paths)
        # s_proj = max(s_proj, self.last_s)

        self.last_s = s_proj

        return s_proj, ey, epsi, psi_r, kappa

    def kappa_at_s(self, s_query):
        # linear interpolation in s
        return float(np.interp(s_query, self.s, self.kappa))

    def v_ref_at_s(self, s_query):
        if self.v_ref is None:
            raise ValueError("v_ref has not been computed. Call compute_speed_profile() first.")
        return float(np.interp(s_query, self.s, self.v_ref))

    def compute_speed_profile(
        self,
        a_lat_max,
        v_min,
        v_max,
        a_accel,
        a_decel,
        kappa_min=1e-3
    ):
        """
        Build offline speed profile from curvature with two-pass acceleration constraints.
        Stores result in self.v_ref.
        """
        if a_lat_max <= 0:
            raise ValueError("a_lat_max must be positive.")
        if v_max <= 0:
            raise ValueError("v_max must be positive.")

        kappa_abs = np.maximum(np.abs(self.kappa), kappa_min)
        v_curve = np.sqrt(a_lat_max / kappa_abs)
        v = np.clip(v_curve, v_min, v_max)

        # Two-pass accel constraints
        ds = np.diff(self.s)
        for i in range(len(ds)):
            v[i + 1] = min(v[i + 1], np.sqrt(v[i]**2 + 2.0 * a_accel * ds[i]))

        for i in range(len(ds) - 1, -1, -1):
            v[i] = min(v[i], np.sqrt(v[i + 1]**2 + 2.0 * a_decel * ds[i]))

        v = np.clip(v, v_min, v_max)
        self.v_ref = v
        np.savetxt("path2d_vref.csv", self.v_ref, delimiter=",", header="v_ref", comments="")
        return self.v_ref

