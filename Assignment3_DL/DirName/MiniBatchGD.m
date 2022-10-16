function [Wstar, bstar, J_train_loss, J_train_cost, Train_acc, J_val_loss, J_val_cost, Val_acc, the_eta] = MiniBatchGD(X_train, Y_train, y_train,X_val, Y_val, y_val, GDParams, W, b, lambda, epsilon)


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

[J_train_loss(1), J_train_cost(1)] = ComputeCost(X_train,Y_train,W,b,lambda,epsilon);
[J_val_loss(1), J_val_cost(1)] = ComputeCost(X_val,Y_val,W,b,lambda,epsilon);
Train_acc(1) = ComputeAccuracy(X_train,y_train,W,b,epsilon);
Val_acc(1) = ComputeAccuracy(X_val,y_val,W,b,epsilon);

iteration = 0;


for i=1:n_epochs
    
    disp(i)

    for j=1:n/n_batch
        j_start = (j-1)*n_batch + 1;
        j_end = j*n_batch;
        Xbatch_train = X_train(:, j_start:j_end); 
        Ybatch_train = Y_train(:, j_start:j_end);
        
        %Shuffling
        rng(iteration);
        Xbatch_train = Xbatch_train(:,randperm(size(Xbatch_train, 2)));
        rng(iteration);
        Ybatch_train = Ybatch_train(:,randperm(size(Ybatch_train, 2)));
        
        % eta_t
        eta = ComputeEta(eta_min, eta_max, n_s, iteration);
        the_eta = [the_eta eta];
        
        % Pass
        [S_batch, ~, Xbatch_layers, P_batch, Mu, Variance] = ForwardPass(Xbatch_train, W, b, epsilon);

        [grad_W,grad_b] = BackwardPass(Xbatch_layers, P_batch, Ybatch_train, S_batch, Mu, Variance, W, lambda, epsilon);
        

        for a=1:length(W)
            
            W{a} = W{a} - eta*grad_W{a};
            b{a} = b{a} - eta*grad_b{a};
            
        end
        
        iteration = iteration + 1;
    end

    [J_train_loss(i+1), J_train_cost(i+1)] = ComputeCost(X_train,Y_train,W,b,lambda,epsilon);
    [J_val_loss(i+1), J_val_cost(i+1)] = ComputeCost(X_val,Y_val,W,b,lambda,epsilon);
    Train_acc(i+1) = ComputeAccuracy(X_train,y_train,W,b,epsilon);
    Val_acc(i+1) = ComputeAccuracy(X_val,y_val,W,b,epsilon);


end

Wstar = W;
bstar = b;

end



