
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


