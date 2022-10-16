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



