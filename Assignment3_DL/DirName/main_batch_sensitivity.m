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
