function index = SynthesizeText(RNN, h0, x0, n, chars_to_keys)

h = h0; % size : m*1
x = x0; % size : K*qqchose
b = RNN.b; % size : m*1
V = RNN.V; % size : K*m
c = RNN.c; % size : *1
W = RNN.W; % size : m*m
U = RNN.U; % size : m*K
index = [];
K = size(x0,1);

for i=1:n % n = length of the sequence to be created

    a = W*h + U*x + b;
    h = tanh(a);
    o = V*h + c;
    p = exp(o)./sum(exp(o));

    cp = cumsum(p);
    a = rand;
    ixs = find(cp-a > 0);
    ii = ixs(1);
    index = [index,ii];
    x = one_hot_encoding(ii, chars_to_keys, K);
end

end


function Y = one_hot_encoding(chars_index_or_str, chars_to_keys, K)

n = length(chars_index_or_str);
Y = zeros(K,n); 

if ischar(chars_index_or_str)
    for i=1:n
        Y(chars_to_keys(chars_index_or_str(i)),i) = 1; 
    end
end

if isfloat(chars_index_or_str)
    for i=1:n
        Y(chars_index_or_str(i),i) = 1;     
    end
end

end

function grads = ComputeGradients(X, Y, intermediary_vectors, P, RNN)

% INPUT

% h : size m*seq_length
% a :  size m*seq_length
% Y_predict : size K*seq_length
% p : size K*seq_length



% Weight parameters
W = RNN.W;
V = RNN.V;
U = RNN.U;
b = RNN.b;



n = size(Y,2);  % n = seq_length
m = size(intermediary_vectors.h,1);

% grad_V
grad_V = 0;

for i=1:n
    grad_o_i = -(Y(:,i) - P(:,i))';
    grad_V = grad_V + grad_o_i'*intermediary_vectors.h(:,i+1)';
end

grads.V = grad_V;


% Compute the_grad_a_t
the_grad_a = {};

grad_o_n = -(Y(:,n) - P(:,n))'; % size 1*K

grad_a_t_plus_1 = grad_o_n*V*(diag(1-(tanh(intermediary_vectors.a(:,n).^2)))); % size 1*m OK


the_grad_a{n} = grad_a_t_plus_1;
for i=n-1:-1:1
    grad_o_i = -(Y(:,i) - P(:,i))';
    grad_h_t = grad_o_i*V + grad_a_t_plus_1*W; %size 1*m
    grad_a_t_plus_1 = grad_h_t*diag(1-(tanh(intermediary_vectors.a(:,i).^2)));
    the_grad_a{i} = grad_a_t_plus_1; % size 1*m
    
end



% grad_W
grad_W = 0;


for i=1:n
    grad_W = grad_W + the_grad_a{i}'*intermediary_vectors.h(:,i)'; %size m*m
end

grads.W = grad_W;

% grad_U
grad_U = 0;
for i=1:n
    grad_U = grad_U + the_grad_a{i}'*X(:,i)';  % size m*K
end

grads.U = grad_U;


% grad_c
grads.c = sum(-(Y-P)')';


% grad_b

the_sum = 0;
for i=1:length(the_grad_a)
    the_sum = the_sum + the_grad_a{i};
end

grad_b = the_sum';
grads.b = grad_b;


for f = fieldnames(grads)'
    grads.(f{1}) = max(min(grads.(f{1}), 5), -5);
end

end


function [loss, the_p, intermediary_vectors] = ForwardPass(X, Y, RNN, h0)

% INPUT

% h0 : size m*1
% X : size K*seq_length
% Y : size K*seq_length
% b : size m*1
% c : size K*1
% V : size K*m


m = size(h0,1);
h = h0;
b = RNN.b;
W = RNN.W;
U = RNN.U;
V = RNN.V;
c = RNN.c;




n = size(X,2);
K = size(X,1);
the_p = zeros(K,n);


% Intermediary vectors
h_t = zeros(m,n);
a_t = zeros(m,n);
h_t(:,1) = h;


for i=1:n

    a = W*h + U*X(:,i) + b;
    a_t(:,i) = a;
    
    h = tanh(a);
    h_t(:,i+1) = h;
    
    o = V*h + c;
    p = exp(o)./sum(exp(o));

    the_p(:,i) = p;

end


intermediary_vectors.h = h_t;
intermediary_vectors.a = a_t;



% Loss

loss = 0;
for i=1:n
    l_cross = -Y(:,i)'*log(the_p(:,i)); % l_cross of the ith (out of n) sample (scalar)
    loss = loss + l_cross;
end



end


function [RNN,M] = BackwardPass(X, Y, intermediary_vectors, RNN, M, P, eta)

grads = ComputeGradients(X, Y, intermediary_vectors, P, RNN);


for f = fieldnames(RNN)'
    M.(f{1}) = M.(f{1}) + grads.(f{1}).^2;
    RNN.(f{1}) = RNN.(f{1}) - eta*(M.(f{1})+eps).^(-0.5).*grads.(f{1});
end


end


function [RNN_best,the_smooth_loss, smooth_loss_best] = SGD_Training(book_data, RNN, M, GDParams, chars_to_keys, keys_to_chars, iter_limit)


m = GDParams(1);
K = GDParams(2);
eta = GDParams(3);
seq_length = GDParams(4);
generation_seq_length = GDParams(5);
n_epochs = GDParams(6);
RNN_best = RNN;
smooth_loss_best = 1000000000;


the_smooth_loss = zeros(iter_limit,1);

length_book = size(book_data,2);
nb_updates = 0;


for i=1:n_epochs
    
    e = 1;
    h_prev = zeros(m,1);

    while e + seq_length-1 < length_book && nb_updates<iter_limit

        X_chars = book_data(e:e+seq_length-1);
        Y_chars = book_data(e+1:e+seq_length);

        % Hot encoding
        X_hot = one_hot_encoding(X_chars, chars_to_keys, K);
        Y_hot = one_hot_encoding(Y_chars, chars_to_keys, K);

        % Forward pass
        [loss, the_p, intermediary_vectors] = ForwardPass(X_hot, Y_hot, RNN, h_prev);

        % Backward pass
        [RNN,M] = BackwardPass(X_hot, Y_hot, intermediary_vectors, RNN, M, the_p, eta);
        
        
        
        if i==1 && e==1
            smooth_loss = loss;
        else
            smooth_loss = .999*smooth_loss + .001*loss;
        end
        
        e = e+seq_length;
        
        
        h_prev = intermediary_vectors.h(:,end);
        
        if smooth_loss < smooth_loss_best
            RNN_best = RNN;
            smooth_loss_best = smooth_loss;
        end
        
        
        if mod(nb_updates,10000) == 0
            disp(' ')
            txt = ['nb_updates: ',num2str(nb_updates),',    ','Smooth loss: ',num2str(smooth_loss)];
            disp(txt)
            disp(' ')
            
        end
        
        
        if mod(nb_updates,10000) == 0
            
            index = SynthesizeText(RNN, h_prev, X_hot(:,1), generation_seq_length, chars_to_keys);
            sentence = [];
            for l=1:length(index)
                char_index = index(l);
                sentence = [sentence keys_to_chars(char_index)];
            end
            
            disp(sentence)
            disp('-----------------------------------')
            
            
        end
        
        nb_updates = nb_updates+1;
        the_smooth_loss(nb_updates) = smooth_loss;
        
        

    end
end


end


%%%%%%%%%% TEST GRAD %%%%%%%%%%%
%{
% Load data
book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c'); 
fclose(fid);

% Unique characters
book_chars = unique(book_data);
K = length(book_chars);

% Mapping
%char_to_ind = containers.Map('KeyType','char','ValueType','int32');
%ind_to_char = containers.Map('KeyType','int32','ValueType','char');
the_keys = 1:K;

the_characters = num2cell(book_chars);
chars_to_keys = containers.Map(the_characters, the_keys); 
keys_to_chars = containers.Map(the_keys, the_characters); 

% Hidden state
m = 5;
eta = 0.1;
seq_length = 25;
sig = 0.01;

% Parameters
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m,K)*sig;
RNN.W = randn(m,m)*sig;
RNN.V = randn(K,m)*sig;


% Forward & Backward Pass
X_chars = book_data(1:seq_length);
Y_chars = book_data(2:seq_length+1);


X_hot = one_hot_encoding(X_chars, chars_to_keys, K);    % Hot encoding
Y_hot = one_hot_encoding(Y_chars, chars_to_keys, K);    % Hot encoding


% Numerical Gradients
h = 1e-4;
num_grads = ComputeGradsNum(X_hot, Y_hot, RNN, h);

% Analytical gradients
h0 = zeros(m,1);
[loss, the_p, intermediary_vectors] = ForwardPass(X_hot, Y_hot, RNN, h0);
grads = ComputeGradients(X_hot, Y_hot, intermediary_vectors, the_p, RNN);



%%%%%% Relative error %%%%%%%


% U
max_error_U = -10000000000000;
for i=1:size(grads.U,1)
    for j=1:size(grads.U,2)

        the_elt_U = abs(grads.U(i,j)-num_grads.U(i,j))/max(abs(grads.U(i,j)),abs(num_grads.U(i,j)));
        
        if max_error_U < the_elt_U
            max_error_U = the_elt_U;
        end
        
    end
end
disp('Max relative error on U')
disp(max_error_U)


% W
max_error_W = -10000000000000;
for i=1:size(grads.W,1)
    for j=1:size(grads.W,2)

        the_elt_W = abs(grads.W(i,j)-num_grads.W(i,j))/max(abs(grads.W(i,j)),abs(num_grads.W(i,j)));
        
        if max_error_W < the_elt_W
            max_error_W = the_elt_W;
        end
        
    end
end
disp('Max relative error on W')
disp(max_error_W)



% b
max_error_V = -10000000000000;
for i=1:size(grads.V,1)
    for j=1:size(grads.V,2)

        the_elt_V = abs(grads.V(i,j)-num_grads.V(i,j))/max(abs(grads.V(i,j)),abs(num_grads.V(i,j)));
        
        if max_error_V < the_elt_V
            max_error_V = the_elt_V;
        end
        
    end
end
disp('Max relative error on V')
disp(max_error_V)


% c
max_error_c = -10000000000000;
for i=1:size(grads.c,1)
    for j=1:size(grads.c,2)

        the_elt_c = abs(grads.c(i,j)-num_grads.c(i,j))/max(abs(grads.c(i,j)),abs(num_grads.c(i,j)));
        
        if max_error_c < the_elt_c
            max_error_c = the_elt_c;
        end
        
    end
end
disp('Max relative error on c')
disp(max_error_c)


% b
max_error_b = -10000000000000;
for i=1:size(grads.b,1)
    for j=1:size(grads.b,2)

        the_elt_b = abs(grads.b(i,j)-num_grads.b(i,j))/max(abs(grads.b(i,j)),abs(num_grads.b(i,j)));
        
        if max_error_b < the_elt_b
            max_error_b = the_elt_b;
        end
        
    end
end
disp('Max relative error on b')
disp(max_error_b)

%}

%%%%%%%%%% MAIN %%%%%%%%%%

% Load data
book_fname = 'data/goblet_book.txt';
fid = fopen(book_fname,'r');
book_data = fscanf(fid,'%c'); 
fclose(fid);

% Unique characters
book_chars = unique(book_data);
K = length(book_chars);

% Mapping
%char_to_ind = containers.Map('KeyType','char','ValueType','int32');
%ind_to_char = containers.Map('KeyType','int32','ValueType','char');
the_keys = 1:K;

the_characters = num2cell(book_chars);
chars_to_keys = containers.Map(the_characters, the_keys); 
keys_to_chars = containers.Map(the_keys, the_characters); 

% Parameters
m = 100;
eta = 0.1;
seq_length = 25;
generation_seq_length = 200;
n_epochs = 10;
iter_limit = 100001;

GDParams = [m K eta seq_length generation_seq_length n_epochs];

% Parameters
sig = 0.01;
RNN.b = zeros(m,1);
RNN.c = zeros(K,1);
RNN.U = randn(m,K)*sig;
RNN.W = randn(m,m)*sig;
RNN.V = randn(K,m)*sig;

for f = fieldnames(RNN)'
    M.(f{1}) = 0;
end


% Training
[RNN,the_smooth_loss, smooth_loss_best] = SGD_Training(book_data, RNN, M, GDParams, chars_to_keys, keys_to_chars, iter_limit);


disp(' ')
txt = ['Best smooth loss = ', num2str(smooth_loss_best)];
disp(txt)
disp (' ')


% Plot

figure
x_plot = 1:iter_limit;
plot(x_plot,the_smooth_loss);

ylabel('Smooth loss','FontSize', 15);
xlabel('Nb of updates','FontSize', 15);



X_chars = book_data(1:seq_length);
X_hot = one_hot_encoding(X_chars, chars_to_keys, K);


h0 = zeros(m,1);
generated_sentence = SynthesizeText(RNN, h0, X_hot(:,1), 1000, chars_to_keys);
sentence = [];
for l=1:length(generated_sentence)
    char_index = generated_sentence(l);
    sentence = [sentence keys_to_chars(char_index)];
end
disp(sentence)




