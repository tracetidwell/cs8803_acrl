function max_clip = max_clipping(target_hover_state, K_ss, idx, H, dt, model)

j = 10;
stable = true;
while stable
    start_ned = [10000; 0; 0];
    rotation_axis = [0; 0; 1];
    rotation_angle = 0; % radians
    start_q = [sin(rotation_angle/2)*rotation_axis; cos(rotation_angle/2)];

    start_state = target_hover_state;
    start_state(idx.ned) = start_ned;
    start_state(idx.q) = start_q;

    clipping_distance =  j;%% your pick
    x(:,1) = start_state;
    for t=1:H
        % control law:
        dx = compute_dx(target_hover_state, x(:,t));
        dx(idx.ned) = max(min(dx(idx.ned), clipping_distance),-clipping_distance);
        delta_u = K_ss*dx;
        % simulate:
        noise_F_T = randn(6,1)*1;
        x(:,t+1) = f_heli(x(:,t), delta_u, dt, model, idx, noise_F_T);
    end
    [p, S] = polyfit(1:2001, x(idx.ned(1), :), 1);
    if S.normr > 100
        stable = false;
    else
        j = j + 10;
    end
end
max_clip = j - 10;
end