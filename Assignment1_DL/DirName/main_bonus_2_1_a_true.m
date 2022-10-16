addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train1, Y_train1, y_train1] = LoadBatch('data_batch_1.mat');
[X_train2, Y_train2, y_train2] = LoadBatch('data_batch_2.mat');
[X_train3, Y_train3, y_train3] = LoadBatch('data_batch_3.mat');
[X_train4, Y_train4, y_train4] = LoadBatch('data_batch_4.mat');
[X_train5, Y_train5, y_train5] = LoadBatch('data_batch_5.mat');

X_val = X_train5(:,9000:end);
X_train5 = X_train5(:,1:8999);
Y_val = Y_train5(:,9000:end);
Y_train5 = Y_train5(:,1:8999);
y_val = y_train5(9000:end);
y_train5 = y_train5(1:8999);

X_train = [X_train1 X_train2 X_train3 X_train4 X_train5];
Y_train = [Y_train1 Y_train2 Y_train3 Y_train4 Y_train5];
y_train = [y_train1; y_train2; y_train3; y_train4; y_train5];



[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');

%X_train size 3072*n_train
%X_test size 3072*n_train
%X_val size 3072*n_train

% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
lambda = 0.1;
eta = 0.001;
n_batch = 100;
n_epochs = 100;
GDParams = [n_batch, eta, n_epochs];

% Initialazing W and b
rng(200);
W = 0.01.*randn(K,d); % W size K*d
b = 0.01.*randn(K,1); % b size K*1

% Training
[Wstar,bstar,J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD_bonus_2_1_a(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);

% Results
disp('Training accuracy: ')
disp(Train_acc(end))
disp('Val accuracy: ')
disp(Val_acc(end))
disp('Test accuracy: ')
disp(ComputeAccuracy(X_test,y_test,Wstar,bstar))

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_loss);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,J_val_loss);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15 );
legend({'Training','Validation'}, 'FontSize', 15 )

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_cost);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,J_val_cost);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);
legend({'Training','Validation'}, 'FontSize', 15 )



figure
x = linspace(1,n_epochs,n_epochs);
plot(x,Train_acc);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,Val_acc);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);
legend({'Training','Validation'},'FontSize', 15)
legend('Location','southeast')





