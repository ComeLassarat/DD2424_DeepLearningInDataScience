function acc = ComputeAccuracy_batch(X,y,W,b,Gamma,Beta,epsilon)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar



[~, ~, ~, P, ~, ~] = ForwardPass_batch(X,W,b,Gamma,Beta, epsilon);

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches
end
