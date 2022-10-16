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

function P = EvaluateClassifier(X,W,b)

% INPUT
% X : size d*n
% W : size K*d
% b : size K*1

% OUTPUT
% P : size K*n

s = W*X + b;   
P = exp(s)./sum(exp(s));

end

function P = EvaluateClassifier_bonus_2_2(X,W,b)

% INPUT
% X : size d*n
% W : size K*d
% b : size K*1

% OUTPUT
% P : size K*n

s = W*X + b;   
P = exp(s)./(exp(s)+1);

end

function acc = ComputeAccuracy(X,y,W,b)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar

P = EvaluateClassifier(X,W,b); % size K*n

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches
end

function [acc, proba_true, proba_false] = ComputeAccuracy_bonus_2_2(X,y,W,b, old_training)

% INPUT
% X : size d*n
% y : size n*1
% W : size K*d
% b : size K*1

% OUTPUT
% acc : scalar

proba_true = [];
proba_false = [];

n = size(X,2);

if old_training == 1
    P = EvaluateClassifier(X,W,b); % size K*n
else
    P = EvaluateClassifier_bonus_2_2(X,W,b); % size K*n
end

[argvalue,argmax] = max(P); % computes the max and argmax for each column --> 1*n vector
acc = sum(argmax == y')/size(X,2);   % 'argmax == y' is a vector with 1 when the component of argmax and y matches


for i=1:n
    if argmax(i) == y(i)
        proba_true = [proba_true argvalue(i)];
    else
        proba_false = [proba_false argvalue(i)];
    end
end

end

function [J_loss, J_cost] = ComputeCost_bonus_2_2(X,Y,W,b,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

n = size(X,2);
K = size(Y,1);

P = EvaluateClassifier_bonus_2_2(X,W,b);
the_sum_total = 0;
for i=1:n
    l_mult_binary_cross = sum((1-Y(:,i)).*log(1-P(:,i)) + Y(:,i).*log(P(:,i)));
    the_sum_total = the_sum_total + (-1/K)*l_mult_binary_cross;

end
J_loss = (1/n)*the_sum_total;
reg = lambda*sum(sum(W.*W));
J_cost = (1/n)*the_sum_total + reg;
end

function [J_loss, J_cost] = ComputeCost(X,Y,W,b,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

P = EvaluateClassifier(X,W,b);
the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end
reg = lambda*sum(sum(W.*W));
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end

function [grad_W,grad_b] = ComputeGradients(X,Y,P,W,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1
% P : size K*n

% OUTPUT

% grad_W : size K*d
% grad_b : size K*1


G_batch = -(Y-P);
n_b = size(X,2);

grad_W = (1/n_b)*G_batch*X' + 2*lambda*W;
grad_b = (1/n_b)*sum(G_batch,2); % sum of G_batch lines

end

function [grad_W,grad_b] = ComputeGradients_bonus_2_2(X,Y,P,W,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1
% P : size K*n

% OUTPUT

% grad_W : size K*d
% grad_b : size K*1


G_batch = -(Y-P);
n_b = size(X,2);
K = size(Y,1);

grad_W = (1/K)*(1/n_b)*G_batch*X' + 2*lambda*W;
grad_b = (1/K)*(1/n_b)*sum(G_batch,2); % sum of G_batch lines

end

function [Wstar, bstar, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda)

n = size(X_train,2);

n_batch = GDParams(1);
eta = GDParams(2);
n_epochs = GDParams(3);

J_train_loss = zeros(n_epochs,1);
J_val_loss = zeros(n_epochs,1);
J_train_cost = zeros(n_epochs,1);
J_val_cost = zeros(n_epochs,1);
Train_acc = zeros(n_epochs,1);
Val_acc = zeros(n_epochs,1);



    
for i=1:n_epochs
    
    

    % Shuffling
    %rng(i)
    %shuffled_X_train = X_train(randperm(size(X_train, 1)), :);
    %rng(i)
    %shuffled_Y_train = Y_train(randperm(size(Y_train, 1)), :);

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        %Xbatch = shuffled_X_train(:, j_start:j_end);
        %Ybatch = shuffled_Y_train(:, j_start:j_end);
        Xbatch = X_train(:, j_start:j_end); 
        Ybatch = Y_train(:, j_start:j_end);

        P = EvaluateClassifier(Xbatch,W,b);

        [grad_W,grad_b] = ComputeGradients(Xbatch,Ybatch,P,W,lambda);

        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end

    %J_train(i) = ComputeCost(shuffled_X_train,shuffled_Y_train,W,b,lambda);
    [J_train_loss(i), J_train_cost(i)] = ComputeCost(X_train,Y_train,W,b,lambda);
    [J_val_loss(i), J_val_cost(i)] = ComputeCost(X_val,Y_val,W,b,lambda);
    Train_acc(i) = ComputeAccuracy(X_train,y_train,W,b);
    Val_acc(i) = ComputeAccuracy(X_val,y_val,W,b);

end


Wstar = W;
bstar = b;

end

function [Wstar, bstar, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD_bonus_2_2(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda)

n = size(X_train,2);

n_batch = GDParams(1);
eta = GDParams(2);
n_epochs = GDParams(3);

J_train_loss = zeros(n_epochs,1);
J_val_loss = zeros(n_epochs,1);
J_train_cost = zeros(n_epochs,1);
J_val_cost = zeros(n_epochs,1);
Train_acc = zeros(n_epochs,1);
Val_acc = zeros(n_epochs,1);



    
for i=1:n_epochs
    
    

    % Shuffling
    %rng(i)
    %shuffled_X_train = X_train(randperm(size(X_train, 1)), :);
    %rng(i)
    %shuffled_Y_train = Y_train(randperm(size(Y_train, 1)), :);

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        inds = j_start:j_end;
        %Xbatch = shuffled_X_train(:, j_start:j_end);
        %Ybatch = shuffled_Y_train(:, j_start:j_end);
        Xbatch = X_train(:, j_start:j_end); 
        Ybatch = Y_train(:, j_start:j_end);

        P = EvaluateClassifier_bonus_2_2(Xbatch,W,b);

        [grad_W,grad_b] = ComputeGradients_bonus_2_2(Xbatch,Ybatch,P,W,lambda);

        W = W - eta*grad_W;
        b = b - eta*grad_b;
    end

    %J_train(i) = ComputeCost(shuffled_X_train,shuffled_Y_train,W,b,lambda);
    [J_train_loss(i), J_train_cost(i)] = ComputeCost_bonus_2_2(X_train,Y_train,W,b,lambda);
    [J_val_loss(i), J_val_cost(i)] = ComputeCost_bonus_2_2(X_val,Y_val,W,b,lambda);
    Train_acc(i) = ComputeAccuracy(X_train,y_train,W,b);
    Val_acc(i) = ComputeAccuracy(X_val,y_val,W,b);

end


Wstar = W;
bstar = b;

end



%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%

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






