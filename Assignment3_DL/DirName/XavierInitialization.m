function [W,b] = XavierInitialization(d,nb_nodes_hidden_layers,K)

rng(100);

nb_hidden = length(nb_nodes_hidden_layers);
W = {};
b = {};

W{1} = double((1/sqrt(d)).*randn(nb_nodes_hidden_layers(1),d));
b{1} = double(zeros(nb_nodes_hidden_layers(1),1)); 


for i=2:nb_hidden
    W{i} = double((1/sqrt(nb_nodes_hidden_layers(i-1))).*randn(nb_nodes_hidden_layers(i),nb_nodes_hidden_layers(i-1)));
    b{i} = double(zeros(nb_nodes_hidden_layers(i),1));
end


% Output layer
W{nb_hidden+1} = double((1/sqrt(nb_nodes_hidden_layers(nb_hidden))).*randn(K,nb_nodes_hidden_layers(nb_hidden)));
b{nb_hidden+1} = double(zeros(K,1));



end