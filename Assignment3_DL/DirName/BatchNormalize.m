function s_norm = BatchNormalize(s,mu,sigma,epsilon)

s_norm = diag((sigma + epsilon).^(-1/2))*(s - mu);

end