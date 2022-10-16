function P = EvaluateClassifier(X,W,b)

% INPUT
% X : size d*n
% W : size K*d
% b : size K*1

% OUTPUT
% P : size K*n

s = W*X + b;   
P = exp(s)./sum(exp(s));

end
