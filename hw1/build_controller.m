function K_ss = build_controller(num_steps)

init_setup;

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

nominal_state = target_hover_state; nominal_inputs = zeros(4,1);
target_state_time_t = target_hover_state; target_state_time_tplus1 = target_hover_state;
magic_factor = 0;
model_bias = zeros(6,1);
simulate_f = @f_heli;

[A, B] = linearized_dynamics(nominal_state, nominal_inputs, ...
	target_state_time_t, target_state_time_tplus1, ...
	simulate_f, dt, model, idx, model_bias, magic_factor, target_state_time_tplus1);

Q = diag(reward.state_multipliers) * dt;
R = diag(reward.input_multipliers) * dt;

Ps = Q;
for i=1:num_steps
	K{i} = (-(B.' * Ps * B + R)) \ B.' * Ps * A;

	Ps = Q + (K{i}.' * R * K{i}) + (A + B * K{i}).' * Ps * (A + B * K{i});
end

K_ss = K{num_steps};