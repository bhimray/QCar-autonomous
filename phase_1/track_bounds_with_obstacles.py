# track_bounds_with_obstacles.py
def track_bounds_with_obstacles(s, track_width, s_obs, ye_obs, obs_w, obs_L):
    """
    Python conversion of MATLAB: track_bounds_with_obstacle.m

    Returns:
        ye_min_s, ye_max_s
    """
    # Base symmetric bounds (no obstacle)
    ye_base_min = -track_width / 2.0
    ye_base_max = track_width / 2.0

    # Obstacle active window in s
    obs_s_min = s_obs - obs_L / 2.0
    obs_s_max = s_obs + obs_L / 2.0

    # bounded error for robustness
    delta_ye = 1e-2

    # Default: with boundedness
    ye_min_s = ye_base_min + delta_ye
    ye_max_s = ye_base_max - delta_ye

    # If we are in the obstacle s-window, shrink one side
    if (s >= obs_s_min) and (s <= obs_s_max):
        # obstacle on the right side -> shrink upper bound
        if ye_obs >= 0:
            ye_max_obs = ye_obs - obs_w / 2.0
            ye_max_s = min(ye_base_max, ye_max_obs)
        else:
            # obstacle on the left side -> shrink lower bound
            ye_min_obs = ye_obs + obs_w / 2.0
            ye_min_s = max(ye_base_min, ye_min_obs)

    return ye_min_s, ye_max_s
