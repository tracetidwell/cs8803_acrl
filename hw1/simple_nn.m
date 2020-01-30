function u = simple_nn(x, w)

%w_init = randn(21*15 + 15 + 15*10 + 10 + 10*4 + 4,1)*.1

input = 21;
d1 = 15;
d2 = 10;
output = 4;

idx1 = input * d1;
idx2 = idx1 + d1;
idx3 = idx2 + d1*d2;
idx4 = idx3 + d2;
idx5 = idx4 + d2*output;

w1 = reshape(w(1:idx1), input, d1);
b1 = reshape(w(idx1+1:idx2), d1, 1);

w2 = reshape(w(idx2+1:idx3), d1, d2);
b2 = reshape(w(idx3+1:idx4), d2, 1);

w3 = reshape(w(idx4+1:idx5), d2, output);
b3 = reshape(w(idx5+1:end), output, 1);

a1 = tanh(w1.' * x + b1);
a2 = tanh(w2.' * a1 + b2);
u = tanh(w3.' * a2 + b3);

end