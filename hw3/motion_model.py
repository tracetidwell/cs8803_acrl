import numpy as np
import matplotlib
import math
from utils import State


def motion_model(params, state_in, action, observed_map, actual_map, goal, times=2):

    if action == -2:

        r_dTheta = -params.d_theta_reverse
        l_dTheta = -params.d_theta_reverse

        R = params.r_radius * r_dTheta
        L = params.l_radius * l_dTheta

        x_out = state_in.x + times * ((R + L) / 2 * math.cos(state_in.theta))
        y_out = state_in.y + times * ((R + L) / 2 * math.sin(state_in.theta))
        theta_out = state_in.theta + times * ((R - L) / params.wb)
        move_count_out = state_in.move_count + 1
        state_out = State(x_out, y_out, theta_out, move_count_out, params)

        flags = 0

    else:

        r_dTheta = params.d_theta_nom + params.d_theta_max_dev * action
        l_dTheta = params.d_theta_nom - params.d_theta_max_dev * action

        R = params.r_radius * r_dTheta
        L = params.l_radius * l_dTheta

        if R == L:
            x_out = state_in.x + times * ((R + L) / 2 * math.cos(state_in.theta))
            y_out = state_in.y + times * ((R + L) / 2 * math.sin(state_in.theta))
        else:
            x_out = state_in.x + params.wb / 2 * (R + L) / (R - L) * (math.sin((R - L) / params.wb + state_in.theta) - math.sin(state_in.theta))
            y_out = state_in.y - params.wb / 2 * (R + L) / (R - L) * (math.cos((R - L) / params.wb + state_in.theta) - math.cos(state_in.theta))

        theta_out = state_in.theta + (R - L) / params.wb
        move_count_out = state_in.move_count + 1
        state_out = State(x_out, y_out, theta_out, move_count_out, params)

        flags = 0

    N, M = actual_map.shape
    x, y = np.meshgrid(range(N), range(M))
    mask = actual_map==0

    poly = matplotlib.path.Path(list(zip(state_out.border[0, :], state_out.border[1, :])))
    collisions = poly.contains_points(list(zip(x[mask], y[mask])))

    if sum(collisions) > 0:
        print('Car has collided')
        state_out = state_in
        state_out.move_count = params.max_move_count
        flags == 2
        return state_out, observed_map, flags

    dists = np.sqrt((x - state_out.x)**2 + (y - state_out.y)**2)
    obs_ind = dists <= params.observation_radius
    observed_map[obs_ind] = actual_map[obs_ind]

    if poly.contains_point((goal.x, goal.y)):
        print('Reached the goal!')
        flags = 1

    return state_out, observed_map, flags
