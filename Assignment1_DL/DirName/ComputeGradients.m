function [grad_W,grad_b] = ComputeGradients(X,Y,P,W,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1
% P : size K*n

% OUTPUT

% grad_W : size K*d
% grad_b : size K*1


G_batch = -(Y-P);
n_b = size(X,2);

grad_W = (1/n_b)*G_batch*X' + 2*lambda*W;
grad_b = (1/n_b)*sum(G_batch,2); % sum of G_batch lines

end