function w_cost = initialize_weights(w_init, K_ss)

init_setup;

H = 250;

aileron_trim = -(model.params.Tx(1)) / model.params.Tx(3);
elevator_trim = -(model.params.Ty(1)) / model.params.Ty(3);
rudder_trim =  -(model.params.Tz(1)) / model.params.Tz(3);

roll_angle_trim = -asin(model.params.Fy(1) / (model.params.m * g) );
collective_trim = (cos(roll_angle_trim)*model.params.Fy(1)-sin(roll_angle_trim)*model.params.Fz(1))/(sin(roll_angle_trim)*model.params.Fz(3));

control_trims = [ aileron_trim ; elevator_trim ; rudder_trim ; collective_trim];
quaternion_trim = [ sin(roll_angle_trim/2) ; 0 ; 0 ; cos(roll_angle_trim/2)];

target_hover_state = [ control_trims; zeros(4,1); zeros(3,1); zeros(3,1); zeros(3,1); quaternion_trim;];

x(:,1) = target_hover_state;
for t=1:H
    dx = compute_dx(target_hover_state, x(:,t));
    v = randn(size(dx,1)-1,1)*.1;
    dx(1:end-1) = dx(1:end-1) + v;
    delta_u = K_ss * dx;
    u_k(:, t) = delta_u;
    noise_F_T = randn(6,1)*.1;
    x(:,t+1) = f_heli(x(:,t), delta_u, dt, model, idx, noise_F_T);
end

%w_init = randn(21*15 + 15 + 15*10 + 10 + 10*4 + 4,1)*.1;
for t=1:H
    dx = compute_dx(target_hover_state, x(:,t));
    v = randn(size(dx,1)-1,1)*.1;
    dx(1:end-1) = dx(1:end-1) + v;
    delta_u = simple_nn(dx, w_init);
    u_nn(:, t) = delta_u;
    noise_F_T = randn(6,1)*.1;
    x(:,t+1) = f_heli(x(:,t), delta_u, dt, model, idx, noise_F_T);
end

w_cost = sum(sum((u_k - u_nn).^2));