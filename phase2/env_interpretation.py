
import numpy as np
from pal.utilities.math import find_overlap, wrap_to_2pi, wrap_to_pi
from scipy import ndimage
from scipy.special import expit, logit

class EnvInterpretation:

    def __init__(self,
            x_min=-4,
            x_max=3,
            y_min=-3,
            y_max=6,
            cellWidth=0.02,
            r_max=5,
            r_res=0.02,
            p_low=0.4,
            p_high=0.6
        ):

        #region define probabilities and their log-odds forms
        self.p_low = p_low
        self.p_prior = 0.5
        self.p_high = p_high
        self.p_sat = 0.001

        self.l_low = logit(self.p_low)
        self.l_prior = logit(self.p_prior)
        self.l_high = logit(self.p_high)
        self.l_min = logit(self.p_sat)
        self.l_max = logit(1-self.p_sat)
        #endregion

        self.init_polar_grid(r_max, r_res)

        self.init_world_map(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            cellWidth=cellWidth
        )
        self.init_patch()

    # ==============  SECTION A - Polar Grid ====================
    def init_polar_grid(self, r_max, r_res):
        # Configuration Parameters for polar grid
        fov = 2*np.pi
        self.phiRes = 1 * np.pi/180
        self.r_max = r_max
        self.r_res = r_res

        # Size of polar patch
        self.mPolarPatch = np.int_(np.ceil(fov / self.phiRes))
        self.nPolarPatch = np.int_(np.floor(self.r_max/self.r_res))

        self.polarPatch = np.zeros(
            shape = (self.mPolarPatch, self.nPolarPatch),
            dtype = np.float32
        )

    def update_polar_grid(self, r):
        # Implement code here to populate the values of self.polarPatch
        # given LiDAR range data 'r'.
        # - r is a 1D list of length self.mPolarPatch
        # - All range measurements are equally self.phiRes radians apart,
        #   starting with 0

        # Implement Your Solution Here
        self.polarPatch[:, :] = self.l_prior

        # 2. Process each LiDAR ray independently
        for phi in range(self.mPolarPatch):

            r_meas = r[phi]

            # --- Case 1: No obstacle detected within range ---
            if np.isnan(r_meas) or r_meas >= self.r_max:
                # self.polarPatch[phi, :] = self.l_low
                continue

            # --- Case 2: Obstacle detected ---
            # Convert measured distance to range-bin index
            hit_bin = int(np.floor(r_meas / self.r_res))

            # Clamp to grid limits (safety)
            hit_bin = min(hit_bin, self.nPolarPatch - 1)

            # Free space up to the obstacle
            if hit_bin > 0:
                self.polarPatch[phi, 0:hit_bin] = self.l_low

            # Occupied cell at the obstacle location
            self.polarPatch[phi, hit_bin] = self.l_high
        pass

    # ==============  SECTION B - Interpolation ====================
    def init_patch(self):
        self.nPatch = np.int_(2*np.ceil(self.r_max/self.cellWidth) + 1)
        self.patch = np.zeros(
            shape = (self.nPatch, self.nPatch),
            dtype = np.float32
        )

    def generate_patch(self, th):
        """
        Convert polarPatch (angle, range) into a local Cartesian patch (x,y),
        rotated by heading th.
        Output: self.patch (log-odds)
        """

        # Start with unknown everywhere
        self.patch[:, :] = self.l_prior

        c = self.nPatch // 2

        # Build grid of (i,j) indices
        ii, jj = np.indices((int(self.nPatch), int(self.nPatch)))

        # Convert indices -> local coordinates (meters), centered at car
        x = (jj - c) * self.cellWidth
        y = (c - ii) * self.cellWidth

        # Rotate into car/LiDAR frame (so we can query polarPatch correctly)
        ct = np.cos(th) + np.pi/2
        st = np.sin(th) + np.pi/2
        x_r =  ct * x + st * y
        y_r = -st * x + ct * y

        # Convert to polar coordinates
        r = np.sqrt(x_r**2 + y_r**2)
        phi = wrap_to_2pi(np.arctan2(y_r, x_r) - np.pi/2)
        phi = np.where(phi < 0, phi + 2*np.pi, phi)

        # Convert to *fractional* indices for interpolation
        phi_f = phi / self.phiRes
        r_f   = r / self.r_res

        # Mask points outside LiDAR range / outside array bounds
        valid = (r_f >= 0) & (r_f <= (self.nPolarPatch - 1)) & \
                (phi_f >= 0) & (phi_f <= (self.mPolarPatch - 1))

        # Interpolate polarPatch at (phi_f, r_f)
        coords = np.vstack((phi_f[valid], r_f[valid]))
        self.patch[valid] = ndimage.map_coordinates(
            self.polarPatch,
            coords,
            order=1,          # bilinear
            mode='nearest'
        )
        pass

    # ==============  SECTION C - Occupancy Grid Update  ====================
    def init_world_map(self,
            x_min = -4,
            x_max = 3,
            y_min = -3,
            y_max = 6,
            cellWidth=0.02
        ):

        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.cellWidth = cellWidth
        self.xLength = x_max - x_min
        self.yLength = y_max - y_min
        self.m = np.int_(np.ceil(self.yLength/self.cellWidth))
        self.n = np.int_(np.ceil(self.xLength/self.cellWidth))

        self.map = np.full(
            shape = (self.m, self.n),
            fill_value = self.l_prior,
            dtype = np.float32
        )

    def xy_to_ij(self, x, y):
        i = np.int_(np.round( (self.y_max - y) / self.cellWidth ))
        j = np.int_(np.round( (x - self.x_min) / self.cellWidth ))
        return i, j

    def updateMap(self, x, y, th, angles, distances):
        # Function created in SECTION A
        self.update_polar_grid(distances)
        
        # Function created in SECTION B
        self.generate_patch(th)
        
        # ----- Section C -----
        # Car position in global map indices
        i0, j0 = self.xy_to_ij(x, y)
        
        # Center index of the patch
        c = self.nPatch // 2
        
        # Calculate the patch's position in the global map
        # Note: patch origin (0,0) is at top-left corner in image coordinates
        i_start = i0 - c  # Top row in global map
        j_start = j0 - c  # Left column in global map
        
        # Find overlapping regions between patch and global map
        aSlice, bSlice = find_overlap(
            self.map,
            self.patch,
            int(i_start),
            int(j_start)
        )
        
        # If there's no overlap, skip update
        if aSlice is None or bSlice is None:
            return
        
        # Extract the overlapping regions
        map_slice = aSlice
        patch_slice = bSlice
        
        # Apply binary Bayes filter update
        # Convert patch values (log-odds) to measurement updates
        # Remove the prior to get just the new evidence
        update = self.patch[patch_slice] - self.l_prior
        
        n_occ = np.sum(update >  1e-6)
        n_free = np.sum(update < -1e-6)
        print("occ:", n_occ, "free:", n_free, "ratio free/occ:", (n_free/(n_occ+1)))
        
       # Apply the update to the map using the original map slice
        self.map[map_slice] += update
        
        # Clip values to prevent numerical instability
        self.map = np.clip(self.map, self.l_min, self.l_max)
        pass
