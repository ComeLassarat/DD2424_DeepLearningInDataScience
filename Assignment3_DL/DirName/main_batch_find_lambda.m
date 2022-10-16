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








