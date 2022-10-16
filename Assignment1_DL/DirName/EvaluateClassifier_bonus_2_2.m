function P = EvaluateClassifier_bonus_2_2(X,W,b)

% INPUT
% X : size d*n
% W : size K*d
% b : size K*1

% OUTPUT
% P : size K*n

s = W*X + b;   
P = exp(s)./(exp(s)+1);

end
