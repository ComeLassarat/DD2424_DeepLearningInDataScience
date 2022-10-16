function index = SynthesizeText(RNN, h0, x0, n, chars_to_keys)

h = h0; % size : m*1
x = x0; % size : K*qqchose
b = RNN.b; % size : m*1
V = RNN.V; % size : K*m
c = RNN.c; % size : *1
W = RNN.W; % size : m*m
U = RNN.U; % size : m*K
index = [];
K = size(x0,1);

for i=1:n % n = length of the sequence to be created

    a = W*h + U*x + b;
    h = tanh(a);
    o = V*h + c;
    p = exp(o)./sum(exp(o));

    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
    index = [index,ii];
    x = one_hot_encoding(ii, chars_to_keys, K);
end




end