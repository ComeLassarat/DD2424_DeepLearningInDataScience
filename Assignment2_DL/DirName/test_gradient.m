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





