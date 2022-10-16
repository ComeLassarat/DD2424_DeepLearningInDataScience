function [Gamma,Beta] = InitializationGammaBeta(W)

Gamma = {};
Beta = {};
rng(100);

for i=1:length(W)-1
    
    %Gamma{i} = randn(size(W{i},1),n_batch);
    Gamma{i} = ones(size(W{i},1),1);
    Beta{i} = zeros(size(W{i},1),1);
    
end

end