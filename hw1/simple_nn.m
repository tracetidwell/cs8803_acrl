function u = simple_nn(x, w)

%w_init = randn(21*15 + 15*10 + 10*4,1)*.1

input = 21;
d1 = 15;
d2 = 10;
output = 4;

idx1 = input * d1;
idx2= idx1 + d1 * d2;

w1 = reshape(w(1:idx1), input, d1);
w2 = reshape(w(idx1+1:idx2), d1, d2);
w3 = reshape(w(idx2+1:end), d2, output);

a1 = relu(w1.' * x);
a2 = relu(w2.' * a1);
u = w3.' * a2;

end