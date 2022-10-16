function [W,b] = Initialization(d,nb_nodes_hidden_layers,K)


% For a k-layer network, there is k-1 hidden layers

% INPUT

% K : number of classes
% nb_nodes_hidden_layers : list of number of nodes in each hidden layers
% d : length of a sample vector

% OUTPUT

% W1 : size nb_nodes_hidden_layers(1)*d
% b1 : size nb_nodes_hidden_layers(1)*1
% W2 : size nb_nodes_hidden_layers(2)*nb_nodes_hidden_layers(1)
% b2 : size nb_nodes_hidden_layers(2)*1
% W3 : size nb_nodes_hidden_layers(3)*nb_nodes_hidden_layers(2)
% b3 : size nb_nodes_hidden_layers(3)*1
% ...
% Wk : size K*nb_nodes_hidden_layers(k)
% bk : size K*1


% Initialazing W
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







