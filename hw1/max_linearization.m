function [max_ned, max_theta] = max_linearization(target_hover_state, K_ss, idx, H, dt, model)

max_ned = zeros(1, 3);
for i=1:3
    j = 10;
    stable = true;
    while stable
        start_ned = [0; 0; 0];
        start_ned(i) = j;
        rotation_axis = [1; 0; 0];
        rotation_angle = 0; % radians
        start_q = [sin(rotation_angle/2)*rotation_axis; cos(rotation_angle/2)];

        start_state = target_hover_state;
        start_state(idx.ned) = start_ned;
        start_state(idx.q) = start_q;

        x(:,1) = start_state;
        for t=1:H
            % control law:
            dx = compute_dx(target_hover_state, x(:,t));
            delta_u = K_ss*dx;
            % simulate:
            x(:,t+1) = f_heli(x(:,t), delta_u, dt, model, idx);
        end
        stable = stabilized(x(:, end), target_hover_state);
        if stable
            j = j + 10;
        end
    end
    max_ned(i) = j;
end

max_theta = zeros(1, 3);
for i=1:3
    stable = true;
    j = 1;
    while stable
        start_ned = [0; 0; 0];
        rotation_axis = [0; 0; 0];
        rotation_axis(i) = 1;
        rotation_angle = (j/8)*pi; % radians
        start_q = [sin(rotation_angle/2)*rotation_axis; cos(rotation_angle/2)];

        start_state = target_hover_state;
        start_state(idx.ned) = start_ned;
        start_state(idx.q) = start_q;

        x(:,1) = start_state;
        for t=1:H
            % control law:
            dx = compute_dx(target_hover_state, x(:,t));
            delta_u = K_ss*dx;
            % simulate:
            x(:,t+1) = f_heli(x(:,t), delta_u, dt, model, idx);
        end
        stable = stabilized(x(:, end), target_hover_state);
        if stable
            j = j + 1;
        end
    end
    max_theta(i) = ((j)/8)*pi;
end
