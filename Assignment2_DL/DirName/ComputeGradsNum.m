function [grad_W, grad_b] = ComputeGradsNum(X, Y, W, b, lambda, h)

grad_W = cell(numel(W), 1);
grad_b = cell(numel(b), 1);

[~,c] = ComputeCost(X, Y, W, b, lambda);
c = c*1000000;

for j=1:length(b)
    grad_b{j} = zeros(size(b{j}));
    
    for i=1:length(b{j})
        b_try = b;
        b_try{j}(i) = b_try{j}(i) + h;
        [ ~, c2] = ComputeCost(X, Y, W, b_try, lambda);
        c2 = c2*1000000;
        grad_b{j}(i) = (c2-c) / h;
    end
end





for j=1:length(W)
    grad_W{j} = zeros(size(W{j}));
    
    for i=1:numel(W{j})   
        W_try = W;
        W_try{j}(i) = W_try{j}(i) + h;
        [ ~, c2] = ComputeCost(X, Y, W_try, b, lambda);
        c2 = c2*1000000;
        grad_W{j}(i) = (c2-c) / h;
    end
end