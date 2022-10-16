function [grad_W, grad_b, grad_gamma, grad_beta] = ComputeGradients_BN(Y, Xp, S, S_hat, P, W, gamma, lambda, mu, v)
    % Init a few useful values
    nl = length(W);
    grad_W = cell(1,nl);
    grad_b = cell(1,nl);
    grad_gamma = cell(1,nl-1);
    grad_beta = cell(1,nl-1);
    
    % Compute G
    G = P-Y;
    
    % First step without the normalization parameters gradients
    k=nl;
    [~, nb] = size(Xp{k});
    grad_W{k} = G*Xp{k}'/nb + 2*lambda*W{k};
    grad_b{k} = G*ones(nb,1)/nb;
    G = W{k}'*G;
    G = G.*(Xp{k}>0);
    
    % Compute intermediary gradients
    for k = nl-1:-1:1
        [~, nb] = size(Xp{k});
        grad_gamma{k} = (G.*S_hat{k})*ones(nb,1)/nb;
        grad_beta{k} = G*ones(nb,1)/nb;
        G = G.*(gamma{k}*ones(1,nb));
        
        % BatchNormBackProp
        s1 = (v{k}+eps).^(-0.5);
        s2 = (v{k}+eps).^(-1.5);
        G1 = G.*(s1*ones(1,nb));
        G2 = G.*(s2*ones(1,nb));
        D = S{k}-mu{k}*ones(1,nb);
        c = (G2.*D)*ones(nb,1);
        G = G1 - (G1*ones(nb,1))*ones(1,nb)/nb - D.*(c*ones(1,nb))/nb;
        
                
        grad_W{k} = G*Xp{k}'/nb + 2*lambda*W{k};
        grad_b{k} = G*ones(nb,1)/nb;
        G = W{k}'*G;
        G = G.*(Xp{k}>0);
    end
end
