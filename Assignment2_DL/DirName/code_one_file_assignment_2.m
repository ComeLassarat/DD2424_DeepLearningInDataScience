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


function [W,b] = Initialization(K,m,d)

% INPUT

% K : number of classes
% m : number of nodes in the hidden layer
% d : length of a sample vector

% OUTPUT

% W1 : size m*d
% b1 : size m*1
% W2 : size K*m
% b2 : size K*1


% Initialazing W
%rng(100);
W1 = double((1/sqrt(d)).*randn(m,d)); % W1 size m*d
W2 = double((1/sqrt(m)).*randn(K,m)); % W2 size K*m


% Initialazing b
%rng(200);
b1 = double(zeros(m,1)); % b1 size m*1
b2 = double(zeros(K,1)); % b2 size K*1


W = {W1,W2};
b = {b1,b2};

end 



function [P,H] = NetworkFunction(Xbatch,W,b)

% INPUT
% X : size d*n
% W1 : size m*d
% b1 : size m*1
% W2 : size K*m
% b2 : size K*1

% OUTPUT
% P : size K*n
% H : size m*n


W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};


s1 = W1*Xbatch + b1;   % size  m*n

H = max(0,s1);   % size m*n
s = W2*H + b2;   % size K*n

P = exp(s)./sum(exp(s));    % size K*n

end

function [J_loss, J_cost] = ComputeCost(X,Y,W,b,lambda)

% INPUT

% X : size d*n
% Y : size K*n
% W : size K*d
% b : size K*1

% OUTPUT

% J : scalar

W1 = W{1};
W2 = W{2};


[P,H] = NetworkFunction(X,W,b);
the_sum = 0;
for i=1:size(X,2)
    l_cross = -Y(:,i)'*log(P(:,i)); % l_cross of the ith (out of n) sample (scalar)
    the_sum = the_sum + l_cross;
end
reg = lambda*sum(sum(W1.*W1)) + lambda*sum(sum(W2.*W2));
J_cost = (1/size(X,2))*the_sum + reg;
J_loss = (1/size(X,2))*the_sum;
end

function eta = ComputeEta(eta_min, eta_max, n_s, iteration)

t = mod(iteration,2*n_s);

if (0 <= t) && (t <= n_s)
    eta = eta_min + (t/n_s)*(eta_max-eta_min);
end

if (n_s < t) && (t <= 2*n_s)
    eta = eta_max - ((t-n_s)/n_s)*(eta_max-eta_min);
end


end

function [grad_W,grad_b] = ComputeGradients(X,Y,P,H,W,lambda)

% INPUT

W1 = W{1};
W2 = W{2};

G_batch = -(Y-P);  % size K*n
n_b = size(X,2);

grad_W2 = (1/n_b)*G_batch*H' +2*lambda*W2;
grad_b2 = (1/n_b)*sum(G_batch,2); % sum of G_batch lines

% Second layer
G_batch = W2'*G_batch;  % size m*n
G_batch(H <= 0) = 0;
%G_batch(H > 0) = 1;

grad_W1 = (1/n_b)*G_batch*X' + 2*lambda*W1;
grad_b1 = (1/n_b)*sum(G_batch,2);

grad_W = {grad_W1,grad_W2};
grad_b = {grad_b1,grad_b2};


end


function [Wstar, bstar, J_train_loss, J_train_cost, Train_acc] = MiniBatchGD_ex_2(X_train, Y_train, y_train, GDParams, W, b, lambda)


W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

n = size(X_train,2);

n_batch = GDParams(1);
eta = GDParams(2);
n_epochs = GDParams(3);

J_train_loss = zeros(n_epochs,1);
J_train_cost = zeros(n_epochs,1);
Train_acc = zeros(n_epochs,1);


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

        [P,H] = NetworkFunction(Xbatch,W,b);

        [grad_W,grad_b] = ComputeGradients(Xbatch,Ybatch,P,H,W,lambda);

        W1 = W1 - eta*grad_W{1};
        W2 = W2 - eta*grad_W{2};
        b1 = b1 - eta*grad_b{1};
        b2 = b2 - eta*grad_b{2};
        
        W={W1,W2};
        b={b1,b2};
        
    end

    %J_train(i) = ComputeCost(shuffled_X_train,shuffled_Y_train,W,b,lambda);
    [J_train_loss(i), J_train_cost(i)] = ComputeCost(X_train,Y_train,W,b,lambda);
    Train_acc(i) = ComputeAccuracy(X_train,y_train,W,b);


end


W1star = W1;
W2star = W2;
b1star = b1;
b2star = b2;

Wstar = {W1star,W2star};
bstar = {b1star,b2star};

end



function [Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD_ex_3_new(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, lambda)


W1 = W{1};
W2 = W{2};
b1 = b{1};
b2 = b{2};

n = size(X_train,2);


eta_min = GDParams(1);
eta_max = GDParams(2);
n_s = GDParams(3);
l = GDParams(4);
n_batch = GDParams(5);

% Parameters of the loops
n_iterations = 2*(l+1)*n_s;
n_epochs = n_iterations/(n/n_batch);


%J_train_loss = zeros(n_iterations,1);
%J_train_cost = zeros(n_iterations,1);
%Train_acc = zeros(n_iterations,1);
%J_val_loss = zeros(n_iterations,1);
%J_val_cost = zeros(n_iterations,1);
%Val_acc = zeros(n_iterations,1);
the_eta = [];

J_train_loss = zeros(n_epochs+1,1);
J_train_cost = zeros(n_epochs+1,1);
Train_acc = zeros(n_epochs+1,1);
J_val_loss = zeros(n_epochs+1,1);
J_val_cost = zeros(n_epochs+1,1);
Val_acc = zeros(n_epochs+1,1);

[J_train_loss(1), J_train_cost(1)] = ComputeCost(X_train,Y_train,W,b,lambda);
[J_val_loss(1), J_val_cost(1)] = ComputeCost(X_val,Y_val,W,b,lambda);
Train_acc(1) = ComputeAccuracy(X_train,y_train,W,b);
Val_acc(1) = ComputeAccuracy(X_val,y_val,W,b);

iteration = 0;

for i=1:n_epochs

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch = X_train(:, j_start:j_end); 
        Ybatch = Y_train(:, j_start:j_end);
        
        
        % eta_t
        eta = ComputeEta(eta_min, eta_max, n_s, iteration);
        the_eta = [the_eta eta];

        [P,H] = NetworkFunction(Xbatch,W,b);

        [grad_W,grad_b] = ComputeGradients(Xbatch,Ybatch,P,H,W,lambda);

        W1 = W1 - eta*grad_W{1};
        W2 = W2 - eta*grad_W{2};
        b1 = b1 - eta*grad_b{1};
        b2 = b2 - eta*grad_b{2};
        
        W={W1,W2};
        b={b1,b2};
        
        %[J_train_loss(iteration + 1), J_train_cost(iteration + 1)] = ComputeCost(X_train,Y_train,W,b,lambda);
        %[J_val_loss(iteration + 1), J_val_cost(iteration + 1)] = ComputeCost(X_val,Y_val,W,b,lambda);
        %Train_acc(iteration + 1) = ComputeAccuracy(X_train,y_train,W,b);
        %Val_acc(iteration + 1) = ComputeAccuracy(X_val,y_val,W,b);
        
        iteration = iteration + 1;
        
    end
        
[J_train_loss(i+1), J_train_cost(i+1)] = ComputeCost(X_train,Y_train,W,b,lambda);
[J_val_loss(i+1), J_val_cost(i+1)] = ComputeCost(X_val,Y_val,W,b,lambda);
Train_acc(i+1) = ComputeAccuracy(X_train,y_train,W,b);
Val_acc(i+1) = ComputeAccuracy(X_val,y_val,W,b);



end



W1star = W1;
W2star = W2;
b1star = b1;
b2star = b2;

Wstar = {W1star,W2star};
bstar = {b1star,b2star};

end





%{
addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
%X_train size 3072*n_train
X_train = X_train(1:9,1);
Y_train = Y_train(:,1);

% Parameters
n_train = size(X_train,2);
K = 10;
d = size(X_train,1);
m = 50;

% Training parameters
lambda = 0.001;
eta = 0.001;

% Initialazing W and b
rng(100);
[W,b] = Initialization(K,m,d);



% Evaluate classifier
[P,H] = NetworkFunction(X_train,W,b);


% Compare gradients
[grad_W, grad_b] = ComputeGradients(X_train,Y_train,P,H,W,lambda);
[grad_W_num, grad_b_num] = ComputeGradsNumSlow(X_train, Y_train, W, b, lambda, 1e-5);

disp('grad_W')
disp(grad_W)
disp('grad_W_num')
disp(grad_W_num)
disp('grad_b')
disp(grad_b)
disp('grad_b_num')
disp(grad_b_num)

disp('Lambda =')
disp(lambda)

disp('Max relative error on W1')
the_max_W1 = -10000000;
the_coord = {1,1};
for i=1:m
    for j=1:d
        the_elt_W1 = abs(grad_W{1}(i,j)-grad_W_num{1}(i,j))/max(abs(grad_W{1}(i,j)),abs(grad_W_num{1}(i,j)));
        %if (abs(grad_W{1}(i,j))> 1e-4 && abs(grad_W_num{1}(i,j))> 1e-4) || (grad_W{1}(i,j) ==0 || grad_W_num{1}(i,j)==0)
            
        if the_max_W1 < the_elt_W1
            the_max_W1 = the_elt_W1;
            the_coord = {i,j};
        end
        %end
    end
end
disp(the_max_W1)


disp('Max relative error on W2')
the_max_W2 = -10000000;
for i=1:K
    for j=1:m
        the_elt_W2 = abs(grad_W{2}(i,j)-grad_W_num{2}(i,j))./max(abs(grad_W{2}(i,j)),abs(grad_W_num{2}(i,j)));
        %if (abs(grad_W{2}(i,j))> 1e-5 && abs(grad_W_num{2}(i,j))> 1e-5) || (grad_W{2}(i,j) ==0 || grad_W_num{2}(i,j)==0)
        if the_max_W2 < the_elt_W2
            the_max_W2 = the_elt_W2;
        end
    end
end
disp(the_max_W2)



disp('Max relative error on b1')
the_max_b1 = -1000000;
for l=1:m
    the_elt_b1 = abs(grad_b{1}(l)-grad_b_num{1}(l))/max(abs(grad_b{1}(l)),abs(grad_b_num{1}(l)));
    if the_max_b1 < the_elt_b1
            the_max_b1 = the_elt_b1;
    end
end
disp(the_max_b1)

disp('Max relative error on b2')
the_max_b2 = -1000000;
for l=1:K
    the_elt_b2 = abs(grad_b{2}(l)-grad_b_num{2}(l))/max(abs(grad_b{2}(l)),abs(grad_b_num{2}(l)));
    if the_max_b2 < the_elt_b2
            the_max_b2 = the_elt_b2;
    end
end
disp(the_max_b2)



%%%%%%%%%%%%%%% Main ex2 %%%%%%%%%%%%%%%%%%

addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
[X_test, Y_test, y_test] = LoadBatch('test_batch.mat');
[X_val, Y_val, y_val] = LoadBatch('data_batch_2.mat');
%X_train size 3072*n_train
%X_test size 3072*n_train
%X_val size 3072*n_train
X_train = X_train(:,1:100);
Y_train = Y_train(:,1:100);
y_train = y_train(1:100);



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
[W,b] = Initialization(K,m,d);
[Wstar, bstar, J_train_loss, J_train_cost, Train_acc] = MiniBatchGD_ex_2(X_train, Y_train, y_train, GDParams, W, b, lambda);

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

%%%%%%%%%%%%%%% Main ex 3 %%%%%%%%%%%%%%%%%%

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



%%%%%%%%%%%%%%% Main ex 4 %%%%%%%%%%%%%%%%%

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
n_s = 800;
l = 2;
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

% Accuracy plot

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



%%%%%%%%%%%%% Main ex 4 part 2 %%%%%%%%%%%%%%%%

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

%}

%%%%%%%%%%%%%% Main ex4 last %%%%%%%%%%%%%%%%


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

X_val = X_train(:,49001:end);
Y_val = Y_train(:,49001:end);
y_val = y_train(49001:end);


X_train = X_train(:,1:49000);
Y_train = Y_train(:,1:49000);
y_train = y_train(1:49000);

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
l = 2;      % = (nb of cycles) -1
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

% Accuracy plot

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


