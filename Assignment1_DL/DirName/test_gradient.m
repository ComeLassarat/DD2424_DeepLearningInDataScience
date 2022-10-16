addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
%X_train size 3072*n_train
X_train = X_train(:,1:2);
Y_train = Y_train(:,1:2);

% Parameters
n_train = size(X_train,2);
K = 10;
d = size(X_train,1);

% Training parameters
lambda = 0.001;
eta = 0.01;

% Initialazing W and b
rng(200);
W = 0.01.*randn(K,d); % W size K*d
b = 0.01.*randn(K,1); % b size K*1

% Evaluate classifier
P = EvaluateClassifier(X_train,W,b);


% Compare gradients
[grad_W, grad_b] = ComputeGradients(X_train,Y_train,P,W,lambda);
[grad_W_num, grad_b_num] = ComputeGradsNumSlow(X_train, Y_train, W, b, lambda, 1e-6);

disp('grad_W')
disp(grad_W(:,1))
disp('grad_W_num')
disp(grad_W_num(:,1))
disp('grad_b')
disp(grad_b)
disp('grad_b_num')
disp(grad_b_num)

disp('Max relative error on W')
the_max_W = -10000000;
for i=1:K
    for j=1:d
        the_elt_W = abs(grad_W(i,1)-grad_W_num(i,1))./max(abs(grad_W(i,1)),abs(grad_W_num(i,1)));
        if the_max_W < the_elt_W
            the_max_W = the_elt_W;
        end
    end
end
disp(the_max_W)

disp('Max relative error on b')
the_max_b = -1000000;
for l=1:K
    the_elt_b = abs(grad_b(l)-grad_b_num(l))/max(abs(grad_b(l)),abs(grad_b_num(l)));
    if the_max_b < the_elt_b
            the_max_b = the_elt_b;
    end
end
disp(the_max_b)





