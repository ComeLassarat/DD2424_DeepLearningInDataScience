function [Sbatch_layers, S_chapeau_layers, Xbatch_layers, P, Mu, Variance] = ForwardPass_batch(Xbatch_train, W, b, Gamma, Beta, epsilon)



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
x = Xbatch_train;
Xbatch_layers{1} = x;

%if isempty(varargin)
    

for i=1:k-1

    s = W{i}*x + b{i};
    Sbatch_layers{i} = s;
    [mu,variance] = ComputeMeanAndVariance(s);
    Mu{i} = mu;
    Variance{i} = variance;


    s_chapeau = BatchNormalize(s,mu,variance,epsilon);
    S_chapeau_layers{i} = s_chapeau;
    s_tilde = s_chapeau.*Gamma{i} + Beta{i};


    x = max(0,s_tilde);
    Xbatch_layers{i+1} = x;           % Xbatch_layers size k: Xbatch_layers{1} = Xbatch(l = 0) = Input. Then Xbatch_layers{2} = Xbatch(l = 1), etc.


end

s = W{k}*x + b{k};
Sbatch_layers{k} = s;

P = exp(s)./sum(exp(s));
    
%end
%{
if length(varargin) == 2 % We have already mean and variance
    
    Mu = varargin{1};
    Variance = varargin{2};
    
     for i=1:k-1

        s = W{i}*x + b{i};
        Sbatch_layers{i} = s;
        


        s_chapeau = BatchNormalize(s,Mu{i},Variance{i},epsilon);
        S_chapeau_layers{i} = s_chapeau;
        s_tilde = s_chapeau.*Gamma{i} + Beta{i};


        x = max(0,s_tilde);
        Xbatch_layers{i+1} = x;           % Xbatch_layers size k: Xbatch_layers{1} = Xbatch(l = 0) = Input. Then Xbatch_layers{2} = Xbatch(l = 1), etc.


     end
     
     s = W{k}*x + b{k};
     Sbatch_layers{k} = s;
     P = exp(s)./sum(exp(s));
    
end
%}

end