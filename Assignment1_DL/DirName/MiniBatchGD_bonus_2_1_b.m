function [Wstar, bstar, J_train_loss, J_train_cost, J_val_loss, J_val_cost, Train_acc, Val_acc] = MiniBatchGD_bonus_2_1_b(X_train, Y_train, y_train, X_val, Y_val, y_val, GDParams, W, b, lambda)

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
        %inds = j_start:j_end;
        %Xbatch = shuffled_X_train(:, j_start:j_end);
        %Ybatch = shuffled_Y_train(:, j_start:j_end);
        Xbatch = X_train(:, j_start:j_end); 
        Ybatch = Y_train(:, j_start:j_end);

        % Flipping
        s = size(Xbatch,2);
        for m=1:s
            im = Xbatch(:,m);
            rdn_nb = rand(1);
            if rdn_nb <= 0.5
                aa = 0:1:31;
                bb = 32:-1:1;
                vv = repmat(32*aa, 32, 1);
                ind_flip = vv(:) + repmat(bb', 32, 1);
                inds_flip = [ind_flip; 1024+ind_flip; 2048+ind_flip];
                im_flipped = im(inds_flip);
                Xbatch(:,m) = im_flipped;
                %variante
                %Xbatch = [Xbatch im_flipped];
                %Ybatch = [Ybatch Ybatch(:,m)];
            end
                
        end
        

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



