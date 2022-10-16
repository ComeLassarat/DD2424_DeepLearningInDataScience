function [grad_W, grad_b, grad_gamma, grad_beta] = BackwardPass_batch(X_batch_layers, P_batch, Y_batch, S_batch_layers, S_chapeau_batch_layers, Mu, Variance, W, Gamma, Beta, lambda, epsilon)


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

for i=k-1:-1:1
    
    % Grad Gamma & Beta
    grad_gamma{i} = (1/n)*sum((G_batch.*S_chapeau_batch_layers{i}),2);
    grad_beta{i} = (1/n)*sum(G_batch,2);
    
    % Propagate gradient for scale and shift
    G_batch = G_batch.*Gamma{i};
    
    % Propagate G_batch through the batch normalization
    G_batch = BatchNormBackPass(G_batch, S_batch_layers{i}, Mu{i}, Variance{i}, epsilon);  % Là les indices sont bons 
    
    % Grad W & b
    grad_W{i} = (1/n)*G_batch*X_batch_layers{i}' + 2*W{i}*lambda;
    grad_b{i} = (1/n)*sum(G_batch,2);
    
    
    % Propagate G_batch to the previous layer
    if i>1
        G_batch = W{i}'*G_batch;
        G_batch(X_batch_layers{i} <= 0) = 0;
        
    end

    
end
%{
grad_gamma{1} = (1/n)*sum((G_batch.*S_chapeau_batch_layers{1}),2);
grad_beta{1} = (1/n)*sum(G_batch,2);


% Propagate gradient for scale and shift
G_batch = G_batch.*Gamma{1};

% Propagate G_batch through the batch normalization
G_batch = BatchNormBackPass(G_batch, S_batch_layers{1}, Mu{1}, Variance{1}, epsilon);  % Là les indices sont bons 


grad_W{1} = (1/n)*G_batch*X_batch_layers{1}' + 2*W{1}*lambda;
grad_b{1} = (1/n)*sum(G_batch,2);
%}
end