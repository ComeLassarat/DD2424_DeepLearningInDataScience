function [grad_W,grad_b] = BackwardPass_test_grad(X_batch_layers, P_batch, Y_batch, S_batch_layers, W, lambda, epsilon)


% INPUT


% Variance : cell size = nb of layers
% Mu : cell size = nb of layers
% Sbatch_layers : cell size = nb of layers
% Sbatch_layers : cell size = nb of layers
% Xbatch_layers : cell size = nb of layers + 1
% P_batch : size K*nb of samples



G_batch = -(Y_batch - P_batch);

k = length(W); 
n = size(P_batch,2);

% For layer k

grad_W{k} = (1/n)*G_batch*X_batch_layers{k}' + 2*W{k}*lambda;
grad_b{k} = (1/n)*sum(G_batch,2);

G_batch = W{k}'*G_batch;
G_batch(X_batch_layers{k} <= 0) = 0;

% Loop

for i=k-1:-1:2
    
    grad_W{i} = (1/n)*G_batch*X_batch_layers{i}' + 2*W{i}*lambda;
    grad_b{i} = (1/n)*sum(G_batch,2);
    
    
    if i>1
        G_batch = W{i}'*G_batch;
        G_batch(X_batch_layers{i} <= 0) = 0;
        
    end

    
end

grad_W{1} = (1/n)*G_batch*X_batch_layers{1}' + 2*W{1}*lambda;
grad_b{1} = (1/n)*sum(G_batch,2);

end