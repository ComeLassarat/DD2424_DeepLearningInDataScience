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