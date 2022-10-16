function [X,Y,y] = LoadBatch(filename)

% OUTPUT
% X : size d*n
% Y : size K*n
% y : size n*1



%Loading data



A = load(filename);
X = double(A.data)/double(255); % Normalizing & X size n*d
X = X'; % X size d*n
n = size(X,2);
y = A.labels+1 ; %coding labels on 1-10
Y = zeros(10,n); %10 = nb of labels
for i=1:n
    Y(y(i),i) = 1;  % +1 because labels are 0 to 9
                             % y is K*n
end

%Pre-processing data
mean_X = mean(X,2);
std_X = std(X,0,2);
X = X - repmat(mean_X, [1, size(X, 2)]);
X = X ./ repmat(std_X, [1, size(X, 2)]);

end


function eta = ComputeEta(eta_min, eta_max, n_s, iteration)

t = mod(iteration,2*n_s);

if (0 <= t) && (t <= n_s)
    eta = eta_min + (t/n_s)*(eta_max-eta_min);
end

if (n_s < t) && (t <= 2*n_s)
    eta = eta_max - ((t-n_s)/n_s)*(eta_max-eta_min);
end


end

function [W,b] = HeInitialization(d,nb_nodes_hidden_layers,K)

rng(107);

nb_hidden = length(nb_nodes_hidden_layers);
W = {};
b = {};

W{1} = double(sqrt(2/d).*randn(nb_nodes_hidden_layers(1),d));
b{1} = double(zeros(nb_nodes_hidden_layers(1),1)); 
%b{1} = double(sqrt(2/d).*randn(nb_nodes_hidden_layers(1),1)); 


for i=2:nb_hidden
    W{i} = double(sqrt(2/nb_nodes_hidden_layers(i-1)).*randn(nb_nodes_hidden_layers(i),nb_nodes_hidden_layers(i-1)));
    b{i} = double(zeros(nb_nodes_hidden_layers(i),1));
    %b{i} = double(sqrt(2/nb_nodes_hidden_layers(i-1)).*randn(nb_nodes_hidden_layers(i),1)); 
end


% Output layer
W{nb_hidden+1} = double(sqrt(2/nb_nodes_hidden_layers(nb_hidden)).*randn(K,nb_nodes_hidden_layers(nb_hidden)));
b{nb_hidden+1} = double(zeros(K,1));
%b{nb_hidden+1} = double(sqrt(2/nb_nodes_hidden_layers(nb_hidden)).*randn(K,1));



end


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

function [J_loss, J_cost] = ComputeCost(X,Y,W,b,lambda,epsilon)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar




[~, ~, ~, P, ~, ~] = ForwardPass(X, W, b, epsilon);
the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end

reg = 0;

for j=1:length(W)
reg = reg + lambda*sum(sum(W{j}.*W{j}));
end
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end

function [J_loss, J_cost] = ComputeCost_batch(X,Y,W,b, Gamma, Beta, lambda, epsilon)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar



[~, ~, ~, P, ~, ~] = ForwardPass_batch(X, W, b, Gamma, Beta, epsilon);

the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end

reg = 0;

for j=1:length(W)
reg = reg + lambda*sum(sum(W{j}.*W{j}));
end
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end


function [J_loss, J_cost] = ComputeCost_batch_for_test_time(X,Y,W,b, Gamma, Beta, lambda, Mu, Variance, epsilon)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar



[~, ~, ~, P] = ForwardPass_batch_for_test_time(X, W, b, Gamma, Beta, Mu, Variance, epsilon);

the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end

reg = 0;

for j=1:length(W)
reg = reg + lambda*sum(sum(W{j}.*W{j}));
end
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end

function [J_loss, J_cost] = ComputeCost_batch_test_grad(X,Y,NetTry,lambda,epsilon)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

W = NetTry.W;
b = NetTry.b;
Gamma = NetTry.gammas;
Beta = NetTry.betas;


[~, ~, ~, P, ~, ~] = ForwardPass_batch(X,W,b,Gamma,Beta,epsilon);

the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end

reg = 0;

for j=1:length(W)
reg = reg + lambda*sum(sum(W{j}.*W{j}));
end
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end

function [J_loss, J_cost] = ComputeCost_test_grad(X,Y,NetTry,lambda,epsilon)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

W = NetTry.W;
b = NetTry.b;


[~, ~, ~, P, ~, ~] = ForwardPass(X,W,b,epsilon);
the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end

reg = 0;

for j=1:length(W)
reg = reg + lambda*sum(sum(W{j}.*W{j}));
end
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end

function [mu,sigma] = ComputeMeanAndVariance(s)

n = size(s,2);

mu = mean(s,2);
sigma = var(s,0,2);
sigma = sigma * ((n-1)/n);
end

function [Sbatch_layers, S_chapeau_layers, Xbatch_layers, P, Mu, Variance] = ForwardPass(Xbatch, W, b, epsilon)

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
    [mu,variance] = ComputeMeanAndVariance(s);
    Mu{i} = mu;
    Variance{i} = variance;
    
    
    s_chapeau = BatchNormalize(s,mu,variance,epsilon); % s_chapeau toujours nul ??
    S_chapeau_layers{i} = s_chapeau;
    
    %x = max(0,s_chapeau);
    x = max(0,s);
    Xbatch_layers{i+1} = x;           % Xbatch_layers size k: Xbatch_layers{1} = Xbatch(l = 0) = Input. Then Xbatch_layers{2} = Xbatch(l = 1), etc.
    
    
end

s = W{k}*x + b{k};
Sbatch_layers{k} = s;

P = exp(s)./sum(exp(s));


end

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

function [Sbatch_layers, S_chapeau_layers, Xbatch_layers, P] = ForwardPass_batch_for_test_time(Xbatch_train, W, b, Gamma, Beta, Mu, Variance, epsilon)



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

% Initialization of the loop
x = Xbatch_train;
Xbatch_layers{1} = x;



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

function s_norm = BatchNormalize(s,mu,sigma,epsilon)

s_norm = diag((sigma + epsilon).^(-1/2))*(s - mu);

end

function new_G_batch = BatchNormBackPass(G_batch, S_batch, mu , variance, epsilon)


n = size(G_batch,2);

sigma_1 = ((variance + epsilon).^(-0.5));
%size(sigma_1)
%size(variance)
%size(G_batch)
sigma_2 = ((variance + epsilon).^(-1.5));


G_1 = G_batch.*sigma_1;
G_2 = G_batch.*sigma_2;
D = S_batch - mu;
c = sum(G_2.*D, 2);
%new_G_batch = G_1 - (1/size(G_1,2))*sum(G_1,2) - (1/size(G_1,2))*D.*c;
new_G_batch = G_1 - (1/n)*sum(G_1,2) - (1/n)*D.*c;

end

function [grad_W,grad_b] = BackwardPass(X_batch_layers, P_batch, Y_batch, S_batch_layers, Mu, Variance, W, lambda, epsilon)


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
    
    
    %G_batch = BatchNormBackPass(G_batch, S_batch_layers{i}, Mu{i}, Variance{i}, epsilon);  % Là les indices sont bons 
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

function acc = ComputeAccuracy(X,y,W,b,epsilon)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar



[~, ~, ~, P, ~, ~] = ForwardPass(X,W,b, epsilon);

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches
end

function acc = ComputeAccuracy_batch(X,y,W,b,Gamma,Beta,epsilon)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar



[~, ~, ~, P, ~, ~] = ForwardPass_batch(X,W,b,Gamma,Beta, epsilon);

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches
end


function acc = ComputeAccuracy_batch_for_test_time(X,y,W,b,Gamma,Beta,Mu,Variance,epsilon)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar



[~, ~, ~, P] = ForwardPass_batch_for_test_time(X,W,b,Gamma,Beta, Mu, Variance, epsilon);

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches
end


function [Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, lambda, epsilon)


n = size(X_train,2);


eta_min = GDParams(1);
eta_max = GDParams(2);
n_s = GDParams(3);
l = GDParams(4);
n_batch = GDParams(5);

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n/n_batch);


%J_train_loss = zeros(n_iterations,1);
%J_train_cost = zeros(n_iterations,1);
%Train_acc = zeros(n_iterations,1);
%J_val_loss = zeros(n_iterations,1);
%J_val_cost = zeros(n_iterations,1);
%Val_acc = zeros(n_iterations,1);
the_eta = [];

J_train_loss = zeros(n_epochs+1,1);
J_train_cost = zeros(n_epochs+1,1);
Train_acc = zeros(n_epochs+1,1);
J_val_loss = zeros(n_epochs+1,1);
J_val_cost = zeros(n_epochs+1,1);
Val_acc = zeros(n_epochs+1,1);

[J_train_loss(1), J_train_cost(1)] = ComputeCost(X_train,Y_train,W,b,lambda,epsilon);
[J_val_loss(1), J_val_cost(1)] = ComputeCost(X_val,Y_val,W,b,lambda,epsilon);
Train_acc(1) = ComputeAccuracy(X_train,y_train,W,b,epsilon);
Val_acc(1) = ComputeAccuracy(X_val,y_val,W,b,epsilon);

iteration = 0;


for i=1:n_epochs
    
    disp(i)

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch_train = X_train(:, j_start:j_end); 
        Ybatch_train = Y_train(:, j_start:j_end);
        
        %Shuffling
        rng(iteration);
        Xbatch_train = Xbatch_train(:,randperm(size(Xbatch_train, 2)));
        rng(iteration);
        Ybatch_train = Ybatch_train(:,randperm(size(Ybatch_train, 2)));
        
        % eta_t
        eta = ComputeEta(eta_min, eta_max, n_s, iteration);
        the_eta = [the_eta eta];
        
        % Pass
        [S_batch, ~, Xbatch_layers, P_batch, Mu, Variance] = ForwardPass(Xbatch_train, W, b, epsilon);

        [grad_W,grad_b] = BackwardPass(Xbatch_layers, P_batch, Ybatch_train, S_batch, Mu, Variance, W, lambda, epsilon);
        

        for a=1:length(W)
            
            W{a} = W{a} - eta*grad_W{a};
            b{a} = b{a} - eta*grad_b{a};
            
        end
        
        iteration = iteration + 1;
    end

    [J_train_loss(i+1), J_train_cost(i+1)] = ComputeCost(X_train,Y_train,W,b,lambda,epsilon);
    [J_val_loss(i+1), J_val_cost(i+1)] = ComputeCost(X_val,Y_val,W,b,lambda,epsilon);
    Train_acc(i+1) = ComputeAccuracy(X_train,y_train,W,b,epsilon);
    Val_acc(i+1) = ComputeAccuracy(X_val,y_val,W,b,epsilon);


end

Wstar = W;
bstar = b;

end



function [Wstar, bstar, Gammastar, Betastar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta, moving_Mu, moving_Sigma] = MiniBatchGD_batch(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, Gamma, Beta, lambda, epsilon)


n = size(X_train,2);

moving_Mu = {};
moving_Sigma = {};

eta_min = GDParams(1);
eta_max = GDParams(2);
n_s = GDParams(3);
l = GDParams(4);
n_batch = GDParams(5);

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n/n_batch);


%J_train_loss = zeros(n_iterations,1);
%J_train_cost = zeros(n_iterations,1);
%Train_acc = zeros(n_iterations,1);
%J_val_loss = zeros(n_iterations,1);
%J_val_cost = zeros(n_iterations,1);
%Val_acc = zeros(n_iterations,1);
the_eta = [];

J_train_loss = zeros(n_epochs+1,1);
J_train_cost = zeros(n_epochs+1,1);
Train_acc = zeros(n_epochs+1,1);
J_val_loss = zeros(n_epochs+1,1);
J_val_cost = zeros(n_epochs+1,1);
Val_acc = zeros(n_epochs+1,1);

[J_train_loss(1), J_train_cost(1)] = ComputeCost_batch(X_train,Y_train,W,b,Gamma,Beta,lambda,epsilon);
[J_val_loss(1), J_val_cost(1)] = ComputeCost_batch(X_val,Y_val,W,b,Gamma,Beta,lambda,epsilon);
Train_acc(1) = ComputeAccuracy_batch(X_train,y_train,W,b,Gamma,Beta,epsilon);
Val_acc(1) = ComputeAccuracy_batch(X_val,y_val,W,b,Gamma,Beta,epsilon);

iteration = 0;

for i=1:n_epochs
    
    disp(i)

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch_train = X_train(:, j_start:j_end); 
        Ybatch_train = Y_train(:, j_start:j_end);
        
        %Shuffling
        rng(iteration);
        Xbatch_train = Xbatch_train(:,randperm(size(Xbatch_train, 2)));
        rng(iteration);
        Ybatch_train = Ybatch_train(:,randperm(size(Ybatch_train, 2)));
        
       
        
        % eta_t
        eta = ComputeEta(eta_min, eta_max, n_s, iteration);
        the_eta = [the_eta eta];
        
        % Pass
        
        [S_batch, S_chapeau_layers, Xbatch_layers, P_batch, Mu, Variance] = ForwardPass_batch(Xbatch_train, W, b, Gamma, Beta, epsilon);

        [grad_W, grad_b, grad_gamma, grad_beta] = BackwardPass_batch(Xbatch_layers, P_batch, Ybatch_train, S_batch, S_chapeau_layers, Mu, Variance, W, Gamma, Beta, lambda, epsilon);

        
        if i==1 && j==1
            
            for t=1:length(Mu)
                moving_Mu{t} = Mu{t};
                moving_Sigma{t} = Variance{t};
            end
           
        else
            
            for t=1:length(Mu)
            
            moving_Mu{t} = 0.7*moving_Mu{t} + (1-0.7)*Mu{t};
            moving_Sigma{t} = 0.7*moving_Sigma{t} + (1-0.7)*Variance{t};
            
            end
            
        end
        
        
        for a=1:length(W)
            
            W{a} = W{a} - eta*grad_W{a};
            b{a} = b{a} - eta*grad_b{a};
            
            
        end
        
        for p=1:length(Gamma)
            
            Gamma{p} = Gamma{p} - eta*grad_gamma{p};
            Beta{p} = Beta{p} - eta*grad_beta{p};
            
        end
        
        iteration = iteration + 1;
    end
    
[J_train_loss(i+1), J_train_cost(i+1)] = ComputeCost_batch_for_test_time(X_train,Y_train,W,b,Gamma,Beta,lambda,moving_Mu,moving_Sigma,epsilon);
[J_val_loss(i+1), J_val_cost(i+1)] = ComputeCost_batch_for_test_time(X_val,Y_val,W,b,Gamma,Beta,lambda,moving_Mu,moving_Sigma,epsilon);
Train_acc(i+1) = ComputeAccuracy_batch_for_test_time(X_train,y_train,W,b,Gamma,Beta,moving_Mu,moving_Sigma,epsilon);
Val_acc(i+1) = ComputeAccuracy_batch_for_test_time(X_val,y_val,W,b,Gamma,Beta,moving_Mu,moving_Sigma,epsilon);


end

Wstar = W;
bstar = b;
Gammastar = Gamma;
Betastar = Beta;

end



function [Wstar, bstar, Gammastar, Betastar, Train_acc, Val_acc, moving_Mu, moving_Sigma] = MiniBatchGD_batch_small(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, Gamma, Beta, lambda, epsilon)


n = size(X_train,2);

moving_Mu = {};
moving_Sigma = {};

eta_min = GDParams(1);
eta_max = GDParams(2);
n_s = GDParams(3);
l = GDParams(4);
n_batch = GDParams(5);

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n/n_batch);



Train_acc = zeros(n_epochs+1,1);
Val_acc = zeros(n_epochs+1,1);

Train_acc(1) = ComputeAccuracy_batch(X_train,y_train,W,b,Gamma,Beta,epsilon);
Val_acc(1) = ComputeAccuracy_batch(X_val,y_val,W,b,Gamma,Beta,epsilon);

iteration = 0;

for i=1:n_epochs
    
    

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch_train = X_train(:, j_start:j_end); 
        Ybatch_train = Y_train(:, j_start:j_end);
        
        %Shuffling
        rng(iteration);
        Xbatch_train = Xbatch_train(:,randperm(size(Xbatch_train, 2)));
        rng(iteration);
        Ybatch_train = Ybatch_train(:,randperm(size(Ybatch_train, 2)));
        
       
        
        % eta_t
        eta = ComputeEta(eta_min, eta_max, n_s, iteration);
        
        % Pass
        
        [S_batch, S_chapeau_layers, Xbatch_layers, P_batch, Mu, Variance] = ForwardPass_batch(Xbatch_train, W, b, Gamma, Beta, epsilon);

        [grad_W, grad_b, grad_gamma, grad_beta] = BackwardPass_batch(Xbatch_layers, P_batch, Ybatch_train, S_batch, S_chapeau_layers, Mu, Variance, W, Gamma, Beta, lambda, epsilon);

        
        if i==1 && j==1
            
            for t=1:length(Mu)
                moving_Mu{t} = Mu{t};
                moving_Sigma{t} = Variance{t};
            end
           
        else
            
            for t=1:length(Mu)
            
            moving_Mu{t} = 0.7*moving_Mu{t} + (1-0.7)*Mu{t};
            moving_Sigma{t} = 0.7*moving_Sigma{t} + (1-0.7)*Variance{t};
            
            end
            
        end
        
        
        for a=1:length(W)
            
            W{a} = W{a} - eta*grad_W{a};
            b{a} = b{a} - eta*grad_b{a};
            
            
        end
        
        for p=1:length(Gamma)
            
            Gamma{p} = Gamma{p} - eta*grad_gamma{p};
            Beta{p} = Beta{p} - eta*grad_beta{p};
            
        end
        
        iteration = iteration + 1;
    end
    
Train_acc(i+1) = ComputeAccuracy_batch_for_test_time(X_train,y_train,W,b,Gamma,Beta,moving_Mu,moving_Sigma,epsilon);
Val_acc(i+1) = ComputeAccuracy_batch_for_test_time(X_val,y_val,W,b,Gamma,Beta,moving_Mu,moving_Sigma,epsilon);


end

Wstar = W;
bstar = b;
Gammastar = Gamma;
Betastar = Beta;

end


%{

%%%%%%%%%%%%% TEST GRAD %%%%%%%%%%%%%

addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
%X_train size 3072*n_train
X_train = X_train(1:9,1:2);
Y_train = Y_train(:,1:2);

% Parameters
n_train = size(X_train,2);
K = 10;
d = size(X_train,1);


% Training parameters
lambda = 0.01;
nb_nodes_hidden_layers = [30,30,30];
epsilon = eps;

% Initialazing W and b
rng(100);
[W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);

NetParams.W = W;
NetParams.b = b;
NetParams.use_bn = 0;


% Compare gradients
[Sbatch_layers, S_chapeau_layers, Xbatch_layers, P_batch] = ForwardPass_test_grad(X_train, W, b, epsilon);

[grad_W,grad_b] = BackwardPass_test_grad(Xbatch_layers, P_batch, Y_train, Sbatch_layers, W, lambda, epsilon);


Grads = ComputeGradsNumSlow(X_train, Y_train, NetParams, lambda, 1e-5);


disp('grad_W')
disp(grad_W)
disp('grad_W_num')
disp(Grads.W)
disp('grad_b')
disp(grad_b)
disp('grad_b_num')
disp(Grads.b)

disp('Lambda =')
disp(lambda)


the_max_W = -10000000;

for z=1:length(Grads.W)
    
    for i=1:size(Grads.W{z},1)
        for j=1:size(Grads.W{z},2)
            %disp({i,j,z})
            
            the_elt_W = abs(grad_W{z}(i,j)-Grads.W{z}(i,j))/max(abs(grad_W{z}(i,j)),abs(Grads.W{z}(i,j)));
 
            if the_max_W < the_elt_W
                the_max_W = the_elt_W;

            end
        end
    end
end
disp('Max relative error on the Ws')
disp(the_max_W)


the_max_b = -1000000;
for p=1:length(Grads.b)
    for l=1:length(Grads.b{p})
        %disp({p,l})
        the_elt_b = abs(grad_b{p}(l)-Grads.b{p}(l))/max(abs(grad_b{p}(l)),abs(Grads.b{p}(l)));
        if the_max_b < the_elt_b
                the_max_b = the_elt_b;
        end
    end
end
disp('Max relative error on the bs')
disp(the_max_b)




%%%%%%%%%%%%%TEST GRAD FOR BATCH NORM%%%%%%%%%%%%%%%


addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
X_train = X_train(1:9,1:16);
Y_train = Y_train(:,1:16);

% Parameters
n_train = size(X_train,2);
K = 10;
d = size(X_train,1);


% Training parameters
lambda = 0;
nb_nodes_hidden_layers = [50,50];
epsilon = eps;

% Initialazing W and b
rng(100);
[W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);
[Gamma,Beta] = InitializationGammaBeta(W);

NetParams.W = W;
NetParams.b = b;
NetParams.gammas = Gamma;
NetParams.betas = Beta;
NetParams.use_bn = 1;


% Compare gradients

[Sbatch_layers, S_chapeau_layers, Xbatch_layers, P_batch, Mu, Variance] = ForwardPass_batch(X_train, W, b, Gamma, Beta, epsilon);

[grad_W, grad_b, grad_gamma, grad_beta] = BackwardPass_batch(Xbatch_layers, P_batch, Y_train, Sbatch_layers, S_chapeau_layers, Mu, Variance, W, Gamma, Beta, lambda, epsilon);


Grads = ComputeGradsNumSlow(X_train, Y_train, NetParams, lambda, 1e-5);

%{
disp('grad_W')
disp(grad_W)
disp('grad_W_num')
disp(Grads.W)
disp('grad_b')
disp(grad_b)
disp('grad_b_num')
disp(Grads.b)

disp('Lambda =')
disp(lambda)
%}



the_max_b = -1000000;
for p=1:length(Grads.b)
    for l=1:length(Grads.b{p})
        
        disp('---------')
        disp({p,l})
        grad_b{p}(l)
        Grads.b{p}(l)
        the_elt_b = abs(grad_b{p}(l)-Grads.b{p}(l))/max(abs(grad_b{p}(l)),abs(Grads.b{p}(l)))
        disp('---------')
        if the_max_b < the_elt_b
                the_max_b = the_elt_b;
        end
    end
end
disp('Max relative error on the bs')
disp(the_max_b)



the_max_W = -10000000;

for z=1:length(Grads.W)
    
    for i=1:size(Grads.W{z},1)
        for j=1:size(Grads.W{z},2)
            
            %disp('---------')
            %disp({z,i,j})
            %grad_W{z}(i,j)
            %Grads.W{z}(i,j)
            
            the_elt_W = abs(grad_W{z}(i,j)-Grads.W{z}(i,j))/max(abs(grad_W{z}(i,j)),abs(Grads.W{z}(i,j)));
            %disp('---------')
            if the_max_W < the_elt_W
                the_max_W = the_elt_W;

            end
        end
    end
end
disp('Max relative error on the Ws')
disp(the_max_W)


the_max_gamma = -1000000;
for p=1:length(Grads.gammas)
    for l=1:length(Grads.gammas{p})
        %disp({p,l})
        the_elt_gamma = abs(grad_gamma{p}(l)-Grads.gammas{p}(l))/max(abs(grad_gamma{p}(l)),abs(Grads.gammas{p}(l)));
        if the_max_gamma < the_elt_gamma
                the_max_gamma = the_elt_gamma;
        end
    end
end
disp('Max relative error on the gammas')
disp(the_max_gamma)


the_max_beta = -1000000;
for p=1:length(Grads.betas)
    for l=1:length(Grads.betas{p})
        
        the_elt_beta = abs(grad_beta{p}(l)-Grads.betas{p}(l))/max(abs(grad_beta{p}(l)),abs(Grads.betas{p}(l)));
        if the_max_beta < the_elt_beta
                the_max_beta = the_elt_beta;
        end
    end
end
disp('Max relative error on the betas')
disp(the_max_beta)



%}

%%%%%%%%%%% MAIN %%%%%%%%%%%%%%
addpath Datasets/cifar-10-batches-mat/;

%{
% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');

%}

% Loading the data
[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1 ;y_train2; y_train3; y_train4; y_train5];

X_val = X_train(:,45001:end);
Y_val = Y_train(:,45001:end);
y_val = y_train(45001:end);


X_train = X_train(:,1:45000);
Y_train = Y_train(:,1:45000);
y_train = y_train(1:45000);

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
epsilon = eps;
lambda = 0.005;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_s = 5*45000/n_batch;
l = 1;  %nb of cycles -1
GDParams = [eta_min, eta_max, n_s, l, n_batch];
nb_nodes_hidden_layers = [50,50];

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n_train/n_batch);


%Initialization of W and b
[W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);
[Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, lambda, epsilon);


% Figures

x=linspace(0,2*(l+1)*n_s,n_epochs+1);
x_eta = linspace(1,2*(l+1)*n_s,2*(l+1)*n_s);

%eta
figure
plot(x_eta,the_eta);
ylabel('eta_t','FontSize', 15);
xlabel('Update step','FontSize', 15);


% Cost
figure
plot(x,J_train_cost);
hold on

plot(x,J_val_cost);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Update step','FontSize', 15);
legend({'Training','Validation'}, 'FontSize', 15 )

%Loss

figure
plot(x,J_train_loss);
hold on

plot(x,J_val_loss);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Update step','FontSize', 15);
legend({'Training','Validation'}, 'FontSize', 15 )

% Accuracy

figure
plot(x,Train_acc);
hold on

plot(x,Val_acc);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Update step','FontSize', 15);
%ylim([0 0.8])
legend({'Training','Validation'}, 'FontSize', 15 )

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy(X_test,y_test,Wstar,bstar,epsilon))



%%%%%%%%%% MAIN BATCH NORM %%%%%%%%%%%%%
addpath Datasets/cifar-10-batches-mat/;

%{
% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');

%}

% Loading the data
[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1 ;y_train2; y_train3; y_train4; y_train5];

X_val = X_train(:,45001:end);
Y_val = Y_train(:,45001:end);
y_val = y_train(45001:end);


X_train = X_train(:,1:45000);
Y_train = Y_train(:,1:45000);
y_train = y_train(1:45000);

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
epsilon = eps;
lambda = 0.005;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_s = 5*45000/n_batch;
l = 1;  %nb of cycles -1
GDParams = [eta_min, eta_max, n_s, l, n_batch];
nb_nodes_hidden_layers = [50,50];

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n_train/n_batch);


%Initialization of W, b, Gamma and Beta
[W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);
[Gamma,Beta] = InitializationGammaBeta(W);




[Wstar, bstar, Gammastar, Betastar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta, moving_Mu, moving_Sigma] = MiniBatchGD_batch(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, Gamma, Beta, lambda, epsilon);


% Figures

x=linspace(0,2*(l+1)*n_s,n_epochs+1);
x_eta = linspace(1,2*(l+1)*n_s,2*(l+1)*n_s);

%eta
figure
plot(x_eta,the_eta);
ylabel('eta_t','FontSize', 15);
xlabel('Update step','FontSize', 15);


% Cost
figure
plot(x,J_train_cost);
hold on

plot(x,J_val_cost);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Update step','FontSize', 15);
%ylim([0 4])
legend({'Training','Validation'}, 'FontSize', 15 )

%Loss

figure
plot(x,J_train_loss);
hold on

plot(x,J_val_loss);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Update step','FontSize', 15);
%ylim([0 3])
legend({'Training','Validation'}, 'FontSize', 15 )

% Accuracy

figure
plot(x,Train_acc);
hold on

plot(x,Val_acc);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Update step','FontSize', 15);
%ylim([0 0.8])
legend({'Training','Validation'}, 'FontSize', 15 )

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy_batch_for_test_time(X_test, y_test, Wstar, bstar, Gammastar, Betastar, moving_Mu, moving_Sigma, epsilon))



%%%%%%%%%%%%%%%%% MAIN TO FIND LAMBDA %%%%%%%%%%%%%%%%%%
addpath Datasets/cifar-10-batches-mat/;

%{
% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');

%}

% Loading the data
[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1 ;y_train2; y_train3; y_train4; y_train5];

X_val = X_train(:,45001:end);
Y_val = Y_train(:,45001:end);
y_val = y_train(45001:end);


X_train = X_train(:,1:45000);
Y_train = Y_train(:,1:45000);
y_train = y_train(1:45000);

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
epsilon = eps;
lambda = 0.005;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_s = 5*45000/n_batch;
l = 1;  %nb of cycles -1
GDParams = [eta_min, eta_max, n_s, l, n_batch];
nb_nodes_hidden_layers = [50,50];

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n_train/n_batch);



%{
% Uniform grid
l_min = -5;
l_max = -1;

lambda_uni_grid = 10.^linspace(l_min,l_max,8);
scores = zeros(8,1);


for elt=1:8
    disp('On est à')
    disp(elt)
    lambda = lambda_uni_grid(elt);
    
    %Initialization of W and b
    rng(100);
    [W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);
    [Gamma,Beta] = InitializationGammaBeta(W);
    [Wstar, bstar, Gammastar, Betastar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta, moving_Mu, moving_Sigma] = MiniBatchGD_batch(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, Gamma, Beta, lambda, epsilon);
    Val_best_acc = max(Val_acc);
    scores(elt) = Val_best_acc;
end

[argvalue,argmax] = maxk(scores,3);

disp('Best scores =')
disp(argvalue)

disp('For lambda = ')
disp(lambda_uni_grid(argmax))

%}

% Random search
l_min = -4;
l_max = -2;


scores_rand = zeros(16,1);
lambda_rand_grid = zeros(16,1);
AM_W = cell(1,16);
AM_b = cell(1,16);
AM_Gamma = cell(1,16);
AM_Beta = cell(1,16);
AM_moving_Mu = cell(1,16);
AM_moving_Sigma = cell(1,16);



for elt=1:16
    disp('On est à')
    disp(elt)
    rng(elt)
    l = l_min + (l_max - l_min)*rand(1, 1);
    lambda = 10^l;
    lambda_rand_grid(elt) = lambda;
    
    %Initialization of W and b
    rng(100);
    [W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);
    [Gamma,Beta] = InitializationGammaBeta(W);
    
    [Wstar, bstar, Gammastar, Betastar, Train_acc, Val_acc, moving_Mu, moving_Sigma] = MiniBatchGD_batch_small(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, Gamma, Beta, lambda, epsilon);
    AM_W{elt} = Wstar;
    AM_b{elt} = bstar;
    AM_Gamma{elt} = Gammastar;
    AM_Beta{elt} = Betastar;
    AM_moving_Mu{elt} = moving_Mu;
    AM_moving_Sigma{elt} = moving_Sigma;
    Val_best_acc = max(Val_acc);
    scores_rand(elt) = Val_best_acc;
end

[argvalue,argmax] = maxk(scores_rand,3);

disp('Best scores =')
disp(argvalue)

disp('For lambda = ')
disp(lambda_rand_grid(argmax))




W_final = AM_W{argmax};
b_final = AM_b{argmax};
Gamma_final = AM_Gamma{argmax};
Beta_final = AM_Beta{argmax};

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy_batch_for_test_time(X_test, y_test, W_final, b_final, Gamma_final, Beta_final, moving_Mu, moving_Sigma, epsilon))



%%%%%%%%%%%%%%%%%% MAIN SENSISTIVITY TEST %%%%%%%%%%%%%%%%%%%%%

addpath Datasets/cifar-10-batches-mat/;

%{
% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');

%}

% Loading the data
[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1 ;y_train2; y_train3; y_train4; y_train5];

X_val = X_train(:,45001:end);
Y_val = Y_train(:,45001:end);
y_val = y_train(45001:end);


X_train = X_train(:,1:45000);
Y_train = Y_train(:,1:45000);
y_train = y_train(1:45000);

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
sig = 1e-4;
epsilon = eps;
lambda = 0.005;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_s = 5*45000/n_batch;
l = 1;  %nb of cycles -1
GDParams = [eta_min, eta_max, n_s, l, n_batch];
nb_nodes_hidden_layers = [50,50];

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n_train/n_batch);


%Initialization of W and b
[W,b] = Initialization_sensitivity(d,nb_nodes_hidden_layers,K,sig);
[Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, lambda, epsilon);


% Figures

x=linspace(0,2*(l+1)*n_s,n_epochs+1);
x_eta = linspace(1,2*(l+1)*n_s,2*(l+1)*n_s);

%eta
figure
plot(x_eta,the_eta);
ylabel('eta_t','FontSize', 15);
xlabel('Update step','FontSize', 15);


% Cost
figure
plot(x,J_train_cost);
hold on

plot(x,J_val_cost);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Update step','FontSize', 15);
legend({'Training','Validation'}, 'FontSize', 15 )

%Loss

figure
plot(x,J_train_loss);
hold on

plot(x,J_val_loss);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Update step','FontSize', 15);
ylim([0 4])
legend({'Training','Validation'}, 'FontSize', 15 )

% Accuracy

figure
plot(x,Train_acc);
hold on

plot(x,Val_acc);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Update step','FontSize', 15);
%ylim([0 0.8])
legend({'Training','Validation'}, 'FontSize', 15 )

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy(X_test,y_test,Wstar,bstar,epsilon))


%%%%%%%%%%%%%%%%%%%%%% MAIN SENSITIVITY TEST BATCH NORM %%%%%%%%%%%%%%%%

addpath Datasets/cifar-10-batches-mat/;

%{
% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');

%}

% Loading the data
[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1 ;y_train2; y_train3; y_train4; y_train5];

X_val = X_train(:,45001:end);
Y_val = Y_train(:,45001:end);
y_val = y_train(45001:end);


X_train = X_train(:,1:45000);
Y_train = Y_train(:,1:45000);
y_train = y_train(1:45000);

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
sig = 1e-4;
epsilon = eps;
lambda = 0.005;
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_s = 5*45000/n_batch;
l = 1;  %nb of cycles -1
GDParams = [eta_min, eta_max, n_s, l, n_batch];
nb_nodes_hidden_layers = [50,50];

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n_train/n_batch);


%Initialization of W, b, Gamma and Beta
[W,b] = Initialization_sensitivity(d,nb_nodes_hidden_layers,K,sig);
[Gamma,Beta] = InitializationGammaBeta(W);




[Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda, epsilon);


% Figures

x=linspace(0,2*(l+1)*n_s,n_epochs+1);
x_eta = linspace(1,2*(l+1)*n_s,2*(l+1)*n_s);

%eta
figure
plot(x_eta,the_eta);
ylabel('eta_t','FontSize', 15);
xlabel('Update step','FontSize', 15);


% Cost
figure
plot(x,J_train_cost);
hold on

plot(x,J_val_cost);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Update step','FontSize', 15);
ylim([0 4])
legend({'Training','Validation'}, 'FontSize', 15 )

%Loss

figure
plot(x,J_train_loss);
hold on

plot(x,J_val_loss);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Update step','FontSize', 15);
ylim([0 3])
legend({'Training','Validation'}, 'FontSize', 15 )

% Accuracy

figure
plot(x,Train_acc);
hold on

plot(x,Val_acc);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Update step','FontSize', 15);
%ylim([0 0.8])
legend({'Training','Validation'}, 'FontSize', 15 )

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy_batch_for_test_time(X_test, y_test, Wstar, bstar, Gammastar, Betastar, moving_Mu, moving_Sigma, epsilon))

