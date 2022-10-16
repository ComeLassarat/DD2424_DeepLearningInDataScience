function [grad_W,grad_b] = ComputeGradients(X,W,b,nb_layers)

G_batch = -(Y_batch - P_batch);
n_b = size(X{end},2);



grad_cost_W_k = (1/n_b)*G_batch*X{end}' +2*lambda*W{end};
grad_cost_b_k = (1/n_b)*sum(G_batch,2);

G_batch = W{end}'*G_batch;
G_batch(X{end} <= 0) = 0;

for l=nb_layers-1:-1:1
    
    grad_cost_gamma_l = (1/n_b)*sum((G_batch.*BatchNormalize(s,mu,sigma,epsilon)),2);
    grad_cost_beta_l = (1/n_b)*sum(G_batch,2);
    
    
    
end


end