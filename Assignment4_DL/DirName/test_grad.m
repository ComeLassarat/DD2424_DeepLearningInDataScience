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





