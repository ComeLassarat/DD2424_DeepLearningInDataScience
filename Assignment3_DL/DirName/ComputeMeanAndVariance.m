function [mu,sigma] = ComputeMeanAndVariance(s)

n = size(s,2);

mu = mean(s,2);
sigma = var(s,0,2);
sigma = sigma * ((n-1)/n);
end