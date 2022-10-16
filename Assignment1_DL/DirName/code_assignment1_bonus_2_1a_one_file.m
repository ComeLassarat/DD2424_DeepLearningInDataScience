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

function [Wstar, bstar, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD_bonus_2_1_a(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda)

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



%%%%%%%%%%%%% MAIN %%%%%%%%%%%%%%%
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





