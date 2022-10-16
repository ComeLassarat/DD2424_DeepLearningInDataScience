function [J_loss, J_cost] = ComputeCost_bonus_2_2(X,Y,W,b,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

n = size(X,2);
K = size(Y,1);

P = EvaluateClassifier_bonus_2_2(X,W,b);
the_sum_total = 0;
for i=1:n
    l_mult_binary_cross = sum((1-Y(:,i)).*log(1-P(:,i)) + Y(:,i).*log(P(:,i)));
    the_sum_total = the_sum_total + (-1/K)*l_mult_binary_cross;

end
J_loss = (1/n)*the_sum_total;
reg = lambda*sum(sum(W.*W));
J_cost = (1/n)*the_sum_total + reg;
end
