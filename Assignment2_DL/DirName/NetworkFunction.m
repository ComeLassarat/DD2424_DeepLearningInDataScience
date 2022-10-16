function [P,H] = NetworkFunction(Xbatch,W,b)

% INPUT
% X : size d*n
% W1 : size m*d
% b1 : size m*1
% W2 : size K*m
% b2 : size K*1

% OUTPUT
% P : size K*n
% H : size m*n


W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};


s1 = W1*Xbatch + b1;   % size  m*n

H = max(0,s1);   % size m*n
s = W2*H + b2;   % size K*n

P = exp(s)./sum(exp(s));    % size K*n

end