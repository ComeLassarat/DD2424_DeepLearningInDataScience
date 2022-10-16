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
lambda = 0.1;
eta = 0.028;
n_batch = 100;
n_epochs = 40;
GDParams = [n_batch, eta, n_epochs];

% Initialazing W and b
rng(200);
W = 0.01.*randn(K,d); % W size K*d
b = 0.01.*randn(K,1); % b size K*1

% Training
[Wstar, bstar, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD_bonus_2_1_d(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda);

% Results
disp('for a starting eta = ')
disp(eta)
disp('Training accuracy: ')
disp(ComputeAccuracy(X_train,y_train,Wstar,bstar))
disp('Val accuracy: ')
disp(ComputeAccuracy(X_val,y_val,Wstar,bstar))
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


% Templates
for i=1:10
    im = reshape(Wstar(i, :), 32, 32, 3);
    s_im{i} = (im - min(im(:))) / (max(im(:)) - min(im(:)));
    s_im{i} = permute(s_im{i}, [2, 1, 3]);
end

figure();
montage(s_im);



