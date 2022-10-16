function [RNN,M] = BackwardPass(X, Y, intermediary_vectors, RNN, M, P, eta)

grads = ComputeGradients(X, Y, intermediary_vectors, P, RNN);


for f = fieldnames(RNN)'
    M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(M.(f{1})+eps).^(-0.5).*grads.(f{1});
end


end