function grads = ComputeGradients(X, Y, intermediary_vectors, P, RNN)

% INPUT

% h : size m*seq_length
% a :  size m*seq_length
% Y_predict : size K*seq_length
% p : size K*seq_length



% Weight parameters
W = RNN.W;
V = RNN.V;
U = RNN.U;
b = RNN.b;



n = size(Y,2);  % n = seq_length
m = size(intermediary_vectors.h,1);

% grad_V
grad_V = 0;

for i=1:n
    grad_o_i = -(Y(:,i) - P(:,i))';
    grad_V = grad_V + grad_o_i'*intermediary_vectors.h(:,i+1)';
end

grads.V = grad_V;


% Compute the_grad_a_t
the_grad_a = {};

grad_o_n = -(Y(:,n) - P(:,n))'; % size 1*K

grad_a_t_plus_1 = grad_o_n*V*(diag(1-(tanh(intermediary_vectors.a(:,n).^2)))); % size 1*m OK


the_grad_a{n} = grad_a_t_plus_1;
for i=n-1:-1:1
    grad_o_i = -(Y(:,i) - P(:,i))';
    grad_h_t = grad_o_i*V + grad_a_t_plus_1*W; %size 1*m
    grad_a_t_plus_1 = grad_h_t*diag(1-(tanh(intermediary_vectors.a(:,i).^2)));
    the_grad_a{i} = grad_a_t_plus_1; % size 1*m
    
end



% grad_W
grad_W = 0;


for i=1:n
    grad_W = grad_W + the_grad_a{i}'*intermediary_vectors.h(:,i)'; %size m*m
end

grads.W = grad_W;

% grad_U
grad_U = 0;
for i=1:n
    grad_U = grad_U + the_grad_a{i}'*X(:,i)';  % size m*K
end

grads.U = grad_U;


% grad_c
grads.c = sum(-(Y-P)')';


% grad_b

the_sum = 0;
for i=1:length(the_grad_a)
    the_sum = the_sum + the_grad_a{i};
end

grad_b = the_sum';
grads.b = grad_b;


for f = fieldnames(grads)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end

end