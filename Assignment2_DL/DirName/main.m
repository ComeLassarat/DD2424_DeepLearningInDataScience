addpath DirName/Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
%X_train size 3072*n_train
%X_test size 3072*n_train
%X_val size 3072*n_train
X_train = X_train(:,100);
Y_train = Y_train(:,100);
y_train = y_train(:,100);



% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);
m = 50; %nb of nodes of the hidden layer

% Training parameters
lambda = 0;
eta = 0.001;
n_batch = 100;
n_epochs = 200;
GDParams = [n_batch, eta, n_epochs];

%Initialization of W and b
[W1,b1,W2,b2] = Initialization(K,m,d);
[W1star, b1star, W2star, b2star, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_cost);
ylabel('Cost','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,Train_acc);
ylabel('Accuracy','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);







