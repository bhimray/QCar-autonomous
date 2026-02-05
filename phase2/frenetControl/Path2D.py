import os
import numpy as np
from pal.utilities.math import wrap_to_pi

class Path2D:
    """
    Waypoints as (N,2). Builds arc-length s_i and heading psi_r(s_i), curvature kappa(s_i).
    """
    def __init__(self, waypoints_xy: np.ndarray):
        assert waypoints_xy.ndim == 2 and waypoints_xy.shape[1] == 2
        self.wp = waypoints_xy
        self.N = self.wp.shape[0]
        self.last_i = 0
        self.last_s = 0.0

        ds = np.linalg.norm(np.diff(self.wp, axis=0), axis=1)
        self.s = np.concatenate(([0.0], np.cumsum(ds)))

        # Smooth the waypoint coordinates (box filter) to reduce curvature noise.
        # This keeps the path shape but damps sharp derivative spikes on discretized data.
        kernel = np.ones(5, dtype=float) / 5.0
        x_s = np.convolve(self.wp[:, 0], kernel, mode="same")
        y_s = np.convolve(self.wp[:, 1], kernel, mode="same")

        # First and second derivatives with respect to arc length s.
        # dx/ds, dy/ds define the tangent direction; d2x/ds2, d2y/ds2 define turning rate.
        dx = np.gradient(x_s, self.s, edge_order=1)
        dy = np.gradient(y_s, self.s, edge_order=1)
        ddx = np.gradient(dx, self.s, edge_order=1)
        ddy = np.gradient(dy, self.s, edge_order=1)

        # Heading psi is the angle of the tangent vector (dx/ds, dy/ds).
        self.psi = np.arctan2(dy, dx)

        # Curvature kappa for a parametric curve (x(s), y(s)):
        # kappa = (x' * y'' - y' * x'') / ( (x'^2 + y'^2)^(3/2) )
        denom = (dx * dx + dy * dy) ** 1.5
        denom = np.where(denom < 1e-12, 1e-12, denom)
        self.kappa = (dx * ddy - dy * ddx) / denom
        self.kappa = self.kappa / 10.0  # scale down curvature for smoother paths

        # Export psi and kappa to separate CSV files in the current working directory.
        out_dir = os.getcwd()
        psi_path = os.path.join(out_dir, "path2d_psi.csv")
        kappa_path = os.path.join(out_dir, "path2d_kappa.csv")
        np.savetxt(psi_path, self.psi, delimiter=",", header="psi", comments="")
        np.savetxt(kappa_path, self.kappa, delimiter=",", header="kappa", comments="")

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
        ds_max = 2.0 * (v_max + 0.1) * Ts  # tune

        s_raw = s_proj
        s_proj = np.clip(s_raw, self.last_s - ds_max, self.last_s + ds_max)

        # enforce no going backward (recommended for racing paths)
        # s_proj = max(s_proj, self.last_s)

        self.last_s = s_proj

        return s_proj, ey, epsi, psi_r, kappa

    def kappa_at_s(self, s_query):
        # linear interpolation in s
        return float(np.interp(s_query, self.s, self.kappa))

