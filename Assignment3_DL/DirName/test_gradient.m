addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
%X_train size 3072*n_train
X_train = X_train(1:9,1:2);
Y_train = Y_train(:,1:2);

% Parameters
n_train = size(X_train,2);
K = 10;
d = size(X_train,1);


% Training parameters
lambda = 0.01;
nb_nodes_hidden_layers = [30,30,30];
epsilon = eps;

% Initialazing W and b
rng(100);
[W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);

NetParams.W = W;
NetParams.b = b;
NetParams.use_bn = 0;


% Compare gradients
[Sbatch_layers, S_chapeau_layers, Xbatch_layers, P_batch] = ForwardPass_test_grad(X_train, W, b, epsilon);

[grad_W,grad_b] = BackwardPass_test_grad(Xbatch_layers, P_batch, Y_train, Sbatch_layers, W, lambda, epsilon);


Grads = ComputeGradsNumSlow(X_train, Y_train, NetParams, lambda, 1e-5);


disp('grad_W')
disp(grad_W)
disp('grad_W_num')
disp(Grads.W)
disp('grad_b')
disp(grad_b)
disp('grad_b_num')
disp(Grads.b)

disp('Lambda =')
disp(lambda)


the_max_W = -10000000;

for z=1:length(Grads.W)
    
    for i=1:size(Grads.W{z},1)
        for j=1:size(Grads.W{z},2)
            %disp({i,j,z})
            
            the_elt_W = abs(grad_W{z}(i,j)-Grads.W{z}(i,j))/max(abs(grad_W{z}(i,j)),abs(Grads.W{z}(i,j)));
 
            if the_max_W < the_elt_W
                the_max_W = the_elt_W;

            end
        end
    end
end
disp('Max relative error on the Ws')
disp(the_max_W)


the_max_b = -1000000;
for p=1:length(Grads.b)
    for l=1:length(Grads.b{p})
        %disp({p,l})
        the_elt_b = abs(grad_b{p}(l)-Grads.b{p}(l))/max(abs(grad_b{p}(l)),abs(Grads.b{p}(l)));
        if the_max_b < the_elt_b
                the_max_b = the_elt_b;
        end
    end
end
disp('Max relative error on the bs')
disp(the_max_b)






