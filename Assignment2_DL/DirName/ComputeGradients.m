function [grad_W,grad_b] = ComputeGradients(X,Y,P,H,W,lambda)

% INPUT

W1 = W{1};
W2 = W{2};

G_batch = -(Y-P);  % size K*n
n_b = size(X,2);

grad_W2 = (1/n_b)*G_batch*H' +2*lambda*W2;
grad_b2 = (1/n_b)*sum(G_batch,2); % sum of G_batch lines

% Second layer
G_batch = W2'*G_batch;  % size m*n
G_batch(H <= 0) = 0;
%G_batch(H > 0) = 1;

grad_W1 = (1/n_b)*G_batch*X' + 2*lambda*W1;
grad_b1 = (1/n_b)*sum(G_batch,2);

grad_W = {grad_W1,grad_W2};
grad_b = {grad_b1,grad_b2};


end
