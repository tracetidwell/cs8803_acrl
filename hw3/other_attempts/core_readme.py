'''While i exist / not in goal'''
while time_running:
    '''Check If we need to replan (map changes with dynamics, or N timesteps)'''
    if past_observed_map != observed_map or timestep > N:

        '''Rough A* in Spatial Domain ONLY - Heuristic to maximize voronoi / Buffer'''
        path = a_star_rough(observed_map, start, goal_node)

        '''Sample Future Point [Snap to goal, if near]'''
        fine_grained_target = snap_to_goal(path[10], goal_node)

        ''' limit search about Rough Tier Plan, then Search '''
        small_map = limit_about(path)
        action_sequence, found = a_star_rough(observed_map, start, fine_grained_target)

        '''If we did not find a solution, lets pick a reversing sequence'''
        if not found:
            action_sequence = choose_best_reverse()

    ''' Perform Given action, check new map, exit if success'''
    action = action_sequence[timestep]; timestep++
    state, observed_map, flags = motion_model(....)
    ...
