function [J_loss, J_cost] = ComputeCost(X,Y,W,b,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

W1 = W{1};
W2 = W{2};


[P,H] = NetworkFunction(X,W,b);
the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end
reg = lambda*sum(sum(W1.*W1)) + lambda*sum(sum(W2.*W2));
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end
