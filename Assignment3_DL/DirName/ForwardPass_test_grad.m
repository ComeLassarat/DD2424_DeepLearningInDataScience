function [Sbatch_layers, S_chapeau_layers, Xbatch_layers, P] = ForwardPass_test_grad(Xbatch,W,b, epsilon)

% OUTPUT

% Variance : cell size = nb of layers
% Mu : cell size = nb of layers
% Sbatch_layers : cell size = nb of layers
% S_chapeau_layers : cell size = nb of layers
% S_chapeau_layers : cell size = nb of layers
% Xbatch_layers : cell size = nb of layers + 1
% P : size K*nb of samples



k = length(W);
Sbatch_layers = {};
Xbatch_layers = {};
S_chapeau_layers = {};
Mu = {};
Variance = {};

% Initialization of the loop
x = Xbatch;
Xbatch_layers{1} = x;

for i=1:k-1
    
    s = W{i}*x + b{i};
    Sbatch_layers{i} = s;

    x = max(0,s);
    Xbatch_layers{i+1} = x;           % Xbatch_layers size k: Xbatch_layers{1} = Xbatch(l = 0) = Input. Then Xbatch_layers{2} = Xbatch(l = 1), etc.
    
    
end

s = W{k}*x + b{k};
Sbatch_layers{k} = s;

P = exp(s)./sum(exp(s));


end