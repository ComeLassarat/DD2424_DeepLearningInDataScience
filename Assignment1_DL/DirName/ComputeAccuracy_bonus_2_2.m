function [acc, proba_true, proba_false] = ComputeAccuracy_bonus_2_2(X,y,W,b, old_training)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar

proba_true = [];
proba_false = [];

n = size(X,2);

if old_training == 1
    P = EvaluateClassifier(X,W,b); % size K*n
else
    P = EvaluateClassifier_bonus_2_2(X,W,b); % size K*n
end

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches


for i=1:n
    if argmax(i) == y(i)
        proba_true = [proba_true argvalue(i)];
    else
        proba_false = [proba_false argvalue(i)];
    end
end

end
