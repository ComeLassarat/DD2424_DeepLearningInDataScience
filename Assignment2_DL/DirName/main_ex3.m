addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
%X_train size 3072*n_train
%X_test size 3072*n_train
%X_val size 3072*n_train
%X_train = X_train(:,1:100);
%Y_train = Y_train(:,1:100);
%y_train = y_train(1:100);

%X_val = X_val(:,1:100);
%Y_val = Y_val(:,1:100);
%y_val = y_val(1:100);

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);
m = 50; %nb of nodes of the hidden layer

% Training parameters
lambda = 0.01;
eta_min = 1e-5;
eta_max = 1e-1;
n_s = 500;
l = 0;
n_batch = 100;
GDParams = [eta_min, eta_max, n_s, l, n_batch];

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n_train/n_batch);

%Initialization of W and b
rng(100);
[W,b] = Initialization(K,m,d);
[Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD_ex_3_new(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, lambda);


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
ylim([0 0.8])
legend({'Training','Validation'}, 'FontSize', 15 )

% Test Accuracy

disp('Test Accuracy = ')
disp(ComputeAccuracy(X_test,y_test,Wstar,bstar))



