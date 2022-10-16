addpath Datasets/cifar-10-batches-mat/;

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
m = 50; %nb of nodes of the hidden layer


% Training parameters
eta_min = 1e-5;
eta_max = 1e-1;
n_batch = 100;
n_s = 2*floor(n_train/n_batch);
l = 1;      % = (nb of cycles) -1
GDParams = [eta_min, eta_max, n_s, l, n_batch];


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
    disp(elt)
    lambda = lambda_uni_grid(elt);
    
    %Initialization of W and b
    rng(100);
    [W,b] = Initialization(K,m,d);
    [Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc] = MiniBatchGD_ex_3_new(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);
    Val_best_acc = max(Val_acc);
    scores(elt) = Val_best_acc;
end

[argvalue,argmax] = maxk(scores,3);

disp('Best scores =')
disp(argvalue)
%0.5266
disp('For lambda = ')
disp(lambda_uni_grid(argmax))
%0.0019


% Result: best score = 0.5266 for lambda = 0.0019


%}

% Then Random search
l_min = -6;
l_max = -4;


scores_rand = zeros(16,1);
lambda_rand_grid = zeros(16,1);
AM_W = cell(1,16);
AM_b = cell(1,16);



for elt=1:16
    disp(elt)
    rng(elt)
    l = l_min + (l_max - l_min)*rand(1, 1);
    lambda = 10^l;
    lambda_rand_grid(elt) = lambda;
    
    %Initialization of W and b
    rng(100);
    [W,b] = Initialization(K,m,d);
    [Wstar, bstar, Train_acc, Val_acc] = MiniBatchGD_ex_3_small(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);
    AM_W{elt} = Wstar;
    AM_b{elt} = bstar;
    Val_best_acc = max(Val_acc);
    scores_rand(elt) = Val_best_acc;
end

[argvalue,argmax] = maxk(scores_rand,3);

disp('Best scores =')
disp(argvalue)
%0.5296
disp('For lambda = ')
disp(lambda_rand_grid(argmax))
%0.0026

% Result: best score = 0.5346 for lambda = 0.0026

W_final = AM_W{argmax};
b_final = AM_b{argmax};

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy(X_test,y_test,W_final,b_final))





