function avg_cost = calculate_cost_nn(w, H, n_iter)

init_setup;

if nargin == 1
    H = 250;
    n_iter = 25;
end

aileron_trim = -(model.params.Tx(1)) / model.params.Tx(3);
elevator_trim = -(model.params.Ty(1)) / model.params.Ty(3);
rudder_trim =  -(model.params.Tz(1)) / model.params.Tz(3);

roll_angle_trim = -asin(model.params.Fy(1) / (model.params.m * g) );
collective_trim = (cos(roll_angle_trim)*model.params.Fy(1)-sin(roll_angle_trim)*model.params.Fz(1))/(sin(roll_angle_trim)*model.params.Fz(3));

control_trims = [ aileron_trim ; elevator_trim ; rudder_trim ; collective_trim];
quaternion_trim = [ sin(roll_angle_trim/2) ; 0 ; 0 ; cos(roll_angle_trim/2)];

target_hover_state = [ control_trims; zeros(4,1); zeros(3,1); zeros(3,1); zeros(3,1); quaternion_trim;];

u_prev_mult = 0; u_delta_prev_mult = 1000;
ned_dot_mult = 1; ned_mult = 1;
pqr_mult = 1; q_mult = 1;
always1state_mult = 0;

reward.state_multipliers = [ u_prev_mult * ones(1,4)   u_delta_prev_mult * ones(1,4)  ned_dot_mult * ones(1,3)  ned_mult*ones(1,3)  ...
	pqr_mult * ones(1,3)  q_mult * ones(1,3)  always1state_mult * ones(1,1)]';
reward.input_multipliers = ones(4,1)*0;

Q = diag(reward.state_multipliers) * dt;
R = diag(reward.input_multipliers) * dt;

all_costs = zeros(1, n_iter);

for i=1:n_iter
    x(:,1) = target_hover_state;
    cost = 0;
    for t=1:H
        dx = compute_dx(target_hover_state, x(:,t));
        % state observation noise:
        v = randn(size(dx,1)-1,1)*.1;
        dx(1:end-1) = dx(1:end-1) + v;
        delta_u = simple_nn(dx, w);
        % simulate:
        noise_F_T = randn(6,1)*.1;
        %u = K * x(:, t);
        cost = cost + x(:, t).' * Q * x(:, t) + delta_u.' * R * delta_u;
        %cost = cost + sqrt(sum((x(:, end) - target_hover_state).^2));
        x(:,t+1) = f_heli(x(:,t), delta_u, dt, model, idx, noise_F_T);
        
    end
    all_costs(i) = cost;
    %all_costs(i) = sqrt(sum(sum((x - target_hover_state).^2)));
end

avg_cost = mean(all_costs);

end