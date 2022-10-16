function [J_loss, J_cost] = ComputeCost_batch_test_grad(X,Y,NetTry,lambda,epsilon)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

W = NetTry.W;
b = NetTry.b;
Gamma = NetTry.gammas;
Beta = NetTry.betas;


[~, ~, ~, P, ~, ~] = ForwardPass_batch(X,W,b,Gamma,Beta,epsilon);

the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end

reg = 0;

for j=1:length(W)
reg = reg + lambda*sum(sum(W{j}.*W{j}));
end
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end
