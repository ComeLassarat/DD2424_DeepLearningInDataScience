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



