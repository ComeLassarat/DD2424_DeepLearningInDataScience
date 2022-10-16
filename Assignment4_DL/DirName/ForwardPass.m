function [loss, the_p, intermediary_vectors] = ForwardPass(X, Y, RNN, h0)

% INPUT

% h0 : size m*1
% X : size K*seq_length
% Y : size K*seq_length
% b : size m*1
% c : size K*1
% V : size K*m


m = size(h0,1);
h = h0;
b = RNN.b;
W = RNN.W;
U = RNN.U;
V = RNN.V;
c = RNN.c;




n = size(X,2);
K = size(X,1);
the_p = zeros(K,n);


% Intermediary vectors
h_t = zeros(m,n);
a_t = zeros(m,n);
h_t(:,1) = h;


for i=1:n

    a = W*h + U*X(:,i) + b;
    a_t(:,i) = a;
    
    h = tanh(a);
    h_t(:,i+1) = h;
    
    o = V*h + c;
    p = exp(o)./sum(exp(o));

    the_p(:,i) = p;

end


intermediary_vectors.h = h_t;
intermediary_vectors.a = a_t;



% Loss

loss = 0;
for i=1:n
    l_cross = -Y(:,i)'*log(the_p(:,i)); % l_cross of the ith (out of n) sample (scalar)
    loss = loss + l_cross;
end



end