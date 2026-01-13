import numpy as np

class PathProgressTracker:
    """
    Tracks monotonic progress s_hat along a 2D waypoint path.
    """

    def __init__(self, waypoints: np.ndarray):
        """
        waypoints: 2xN array [x; y]
        """
        assert waypoints.shape[0] == 2
        self.waypoints = waypoints
        self.s = self._compute_arc_length(waypoints)
        self.s_hat = 0.0

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def reset_with_pose(self, x: float, y: float):
        """
        Call this when a NEW path is loaded (replanning).
        """
        s_meas, _, _ = self.project(x, y)
        self.s_hat = s_meas

    def update(self, x: float, y: float):
        """
        Call this every control cycle.
        """
        s_meas, d, idx = self.project(x, y)
        self.s_hat = max(self.s_hat, s_meas)
        return self.s_hat, s_meas, d, idx

    def get_reference_window(self, lookahead: float):
        """
        Returns indices of waypoints in [s_hat, s_hat + lookahead]
        """
        s_start = self.s_hat
        s_end = self.s_hat + lookahead
        mask = (self.s >= s_start) & (self.s <= s_end)
        return np.where(mask)[0]

    # --------------------------------------------------
    # Geometry
    # --------------------------------------------------

    def project(self, x: float, y: float):
        """
        Projects (x,y) onto the waypoint polyline.

        Returns:
            s_proj  : arc-length at projection
            d       : signed lateral distance
            seg_idx : segment index
        """
        p = np.array([x, y])

        best_dist = float("inf")
        best_s = 0.0
        best_d = 0.0
        best_i = 0

        for i in range(len(self.s) - 1):
            p0 = self.waypoints[:, i]
            p1 = self.waypoints[:, i + 1]
            v = p1 - p0
            w = p - p0

            seg_len2 = np.dot(v, v)
            if seg_len2 < 1e-9:
                continue

            t = np.clip(np.dot(w, v) / seg_len2, 0.0, 1.0)
            proj = p0 + t * v

            dist = np.linalg.norm(p - proj)
            if dist < best_dist:
                best_dist = dist
                best_s = self.s[i] + t * np.linalg.norm(v)

                # signed lateral error (2D cross product)
                cross = v[0] * (p[1] - proj[1]) - v[1] * (p[0] - proj[0])
                best_d = np.sign(cross) * dist
                best_i = i

        return best_s, best_d, best_i

    @staticmethod
    def _compute_arc_length(waypoints: np.ndarray):
        s = np.zeros(waypoints.shape[1])
        for i in range(1, waypoints.shape[1]):
            dx = waypoints[0, i] - waypoints[0, i - 1]
            dy = waypoints[1, i] - waypoints[1, i - 1]
            s[i] = s[i - 1] + np.hypot(dx, dy)
        return s
