addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
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
lambda = 0.004;
eta = 0.013;
n_batch = 100;
n_epochs = 40;
GDParams = [n_batch, eta, n_epochs];

% Initialazing W and b
rng(200);
W = 0.01.*randn(K,d); % W size K*d
b = 0.01.*randn(K,1); % b size K*1

% New training
[Wstar_new_training, bstar_new_training, J_train_loss_new_training, J_train_cost_new_training, J_val_loss_new_training, J_val_cost_new_training, Train_acc_new_training, Val_acc_new_training] = MiniBatchGD_bonus_2_2(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);
%Old training
[Wstar, bstar, ~ , ~ , ~ , ~ , ~ , ~ ] = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);

% Results
disp('eta')
disp(eta)
disp('lambda')
disp(lambda)
disp('Training accuracy: ')
[acc_training, ~, ~] = ComputeAccuracy_bonus_2_2(X_train, y_train, Wstar_new_training, bstar_new_training, 0);
disp(acc_training)
disp('Val accuracy: ')
[acc_val, ~, ~] = ComputeAccuracy_bonus_2_2(X_val, y_val, Wstar_new_training, bstar_new_training, 0);
disp(acc_val)
disp('Test accuracy: ')
[acc_test, proba_true_test, proba_false_test] = ComputeAccuracy_bonus_2_2(X_test, y_test, Wstar_new_training, bstar_new_training, 0);
disp(acc_test)

% Results for old training
[~ , proba_true_test_old_test, proba_false_test_old_test] = ComputeAccuracy_bonus_2_2(X_test, y_test, Wstar, bstar, 1);

%Histogram

%True
figure
h1 = histogram(proba_true_test);
hold on
h2 = histogram(proba_true_test_old_test);
xlabel('Probability','FontSize', 15);
ylabel('Number of samples','FontSize', 15 );
title('Probabilities of samples well classified','FontSize', 15)
legend({'New training (Sigmoid)','Old training (Softmax)'}, 'FontSize', 15 )

%False
figure
h3 = histogram(proba_false_test);
hold on
h4 = histogram(proba_false_test_old_test);
xlabel('Probability','FontSize', 15);
ylabel('Number of samples','FontSize', 15 );
title('Probabilities of samples misclassified','FontSize', 15)
legend({'New training (Sigmoid)','Old training (Softmax)'}, 'FontSize', 15 )


figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_loss_new_training);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,J_val_loss_new_training);

hold off
ylabel('Loss','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15 );
legend({'Training','Validation'}, 'FontSize', 15 )

figure
x = linspace(1,n_epochs,n_epochs);
plot(x,J_train_cost_new_training);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,J_val_cost_new_training);

hold off
ylabel('Cost','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);
legend({'Training','Validation'}, 'FontSize', 15 )


figure
x = linspace(1,n_epochs,n_epochs);
plot(x,Train_acc_new_training);

hold on
x = linspace(1,n_epochs,n_epochs);
plot(x,Val_acc_new_training);

hold off
ylabel('Accuracy','FontSize', 15);
xlabel('Nb of epochs','FontSize', 15);
legend({'Training','Validation'},'FontSize', 15)


% Templates
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

figure();
montage(s_im);



