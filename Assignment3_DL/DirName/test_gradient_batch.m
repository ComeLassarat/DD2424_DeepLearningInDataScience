addpath Datasets/cifar-10-batches-mat/;

% Loading the data
[X_train, Y_train, y_train] = LoadBatch('data_batch_1.mat');
X_train = X_train(1:9,1:16);
Y_train = Y_train(:,1:16);

% Parameters
n_train = size(X_train,2);
K = 10;
d = size(X_train,1);


% Training parameters
lambda = 0;
nb_nodes_hidden_layers = [50,50];
epsilon = eps;

% Initialazing W and b
rng(100);
[W,b] = HeInitialization(d,nb_nodes_hidden_layers,K);
[Gamma,Beta] = InitializationGammaBeta(W);

NetParams.W = W;
NetParams.b = b;
NetParams.gammas = Gamma;
NetParams.betas = Beta;
NetParams.use_bn = 1;


% Compare gradients

[Sbatch_layers, S_chapeau_layers, Xbatch_layers, P_batch, Mu, Variance] = ForwardPass_batch(X_train, W, b, Gamma, Beta, epsilon);

[grad_W, grad_b, grad_gamma, grad_beta] = BackwardPass_batch(Xbatch_layers, P_batch, Y_train, Sbatch_layers, S_chapeau_layers, Mu, Variance, W, Gamma, Beta, lambda, epsilon);


Grads = ComputeGradsNumSlow(X_train, Y_train, NetParams, lambda, 1e-5);

%{
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
%}



the_max_b = -1000000;
for p=1:length(Grads.b)
    for l=1:length(Grads.b{p})
        
        disp('---------')
        disp({p,l})
        grad_b{p}(l)
        Grads.b{p}(l)
        the_elt_b = abs(grad_b{p}(l)-Grads.b{p}(l))/max(abs(grad_b{p}(l)),abs(Grads.b{p}(l)))
        disp('---------')
        if the_max_b < the_elt_b
                the_max_b = the_elt_b;
        end
    end
end
disp('Max relative error on the bs')
disp(the_max_b)



the_max_W = -10000000;

for z=1:length(Grads.W)
    
    for i=1:size(Grads.W{z},1)
        for j=1:size(Grads.W{z},2)
            
            %disp('---------')
            %disp({z,i,j})
            %grad_W{z}(i,j)
            %Grads.W{z}(i,j)
            
            the_elt_W = abs(grad_W{z}(i,j)-Grads.W{z}(i,j))/max(abs(grad_W{z}(i,j)),abs(Grads.W{z}(i,j)));
            %disp('---------')
            if the_max_W < the_elt_W
                the_max_W = the_elt_W;

            end
        end
    end
end
disp('Max relative error on the Ws')
disp(the_max_W)


the_max_gamma = -1000000;
for p=1:length(Grads.gammas)
    for l=1:length(Grads.gammas{p})
        %disp({p,l})
        the_elt_gamma = abs(grad_gamma{p}(l)-Grads.gammas{p}(l))/max(abs(grad_gamma{p}(l)),abs(Grads.gammas{p}(l)));
        if the_max_gamma < the_elt_gamma
                the_max_gamma = the_elt_gamma;
        end
    end
end
disp('Max relative error on the gammas')
disp(the_max_gamma)


the_max_beta = -1000000;
for p=1:length(Grads.betas)
    for l=1:length(Grads.betas{p})
        
        the_elt_beta = abs(grad_beta{p}(l)-Grads.betas{p}(l))/max(abs(grad_beta{p}(l)),abs(Grads.betas{p}(l)));
        if the_max_beta < the_elt_beta
                the_max_beta = the_elt_beta;
        end
    end
end
disp('Max relative error on the betas')
disp(the_max_beta)




