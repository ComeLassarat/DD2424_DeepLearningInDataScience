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


% Parameters
n_train = size(X_train,2);
n_test = size(X_test,2);
n_val = size(X_val,2);
K = 10;
d = size(X_train,1);

% Training parameters
lambda = 0.0025;
eta = 0.001;
n_batch = 100;
n_epochs = 40;
GDParams = [n_batch, eta, n_epochs];



for t=1:10
    % Initialazing W and b
    rng(t+33);
    W = 0.01.*randn(K,d); % W size K*d
    b = 0.01.*randn(K,1); % b size K*1
    
    J_train_loss_mean = zeros(n_epochs,1);
    J_train_cost_mean = zeros(n_epochs,1);
    J_val_loss_mean = zeros(n_epochs,1);
    J_val_cost_mean = zeros(n_epochs,1);
    Train_acc_mean = zeros(n_epochs,1);
    Val_acc_mean = zeros(n_epochs,1);
    Test_acc_mean_last = 0;
    
    % Training
    [Wstar, bstar, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD_bonus_2_1_a(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);
    
    J_train_loss_mean = J_train_loss_mean + J_train_loss;
    J_train_cost_mean = J_train_cost_mean + J_train_cost;
    J_val_loss_mean = J_val_loss_mean + J_val_loss;
    J_val_cost_mean = J_val_cost_mean + J_val_cost;
    Train_acc_mean = Train_acc_mean + Train_acc;
    Val_acc_mean = Val_acc_mean + Val_acc;
    Test_acc_mean_last = Test_acc_mean_last + ComputeAccuracy(X_test, y_test, Wstar, bstar);
end

J_train_loss_mean = J_train_loss_mean/10;
J_train_cost_mean = J_train_cost_mean/10;
J_val_loss_mean = J_val_loss_mean/10;
J_val_cost_mean = J_val_cost_mean/10;
Train_acc_mean = Train_acc_mean/10;
Val_acc_mean = Val_acc_mean/10;
Test_acc_mean_last = Test_acc_mean_last/10;
    
% Results
disp('Mean training accuracy: ')
disp(Train_acc_mean(end))
disp('Val accuracy: ')
disp(Val_acc_mean(end))
disp('Test accuracy: ')
disp(Test_acc_mean_last)

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_loss_mean);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,J_val_loss_mean);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15 );
legend({'Training','Validation'}, 'FontSize', 15 )

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_cost_mean);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,J_val_cost_mean);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);
legend({'Training','Validation'}, 'FontSize', 15 )



figure
x = linspace(1,n_epochs,n_epochs);
plot(x,Train_acc_mean);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,Val_acc_mean);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);
legend({'Training','Validation'},'FontSize', 15)
legend('Location','southeast')





