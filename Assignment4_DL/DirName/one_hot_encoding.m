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