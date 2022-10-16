function [W,b] = Initialization(K,m,d)

% INPUT

% K : number of classes
% m : number of nodes in the hidden layer
% d : length of a sample vector

% OUTPUT

% W1 : size m*d
% b1 : size m*1
% W2 : size K*m
% b2 : size K*1


% Initialazing W
%rng(100);
W1 = double((1/sqrt(d)).*randn(m,d)); % W1 size m*d
W2 = double((1/sqrt(m)).*randn(K,m)); % W2 size K*m


% Initialazing b
%rng(200);
b1 = double(zeros(m,1)); % b1 size m*1
b2 = double(zeros(K,1)); % b2 size K*1


W = {W1,W2};
b = {b1,b2};

end 







