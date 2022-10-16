function new_G_batch = BatchNormBackPass(G_batch, S_batch, mu , variance, epsilon)


n = size(G_batch,2);

sigma_1 = ((variance + epsilon).^(-0.5));
%size(sigma_1)
%size(variance)
%size(G_batch)
sigma_2 = ((variance + epsilon).^(-1.5));


G_1 = G_batch.*sigma_1;
G_2 = G_batch.*sigma_2;
D = S_batch - mu;
c = sum(G_2.*D, 2);
%new_G_batch = G_1 - (1/size(G_1,2))*sum(G_1,2) - (1/size(G_1,2))*D.*c;
new_G_batch = G_1 - (1/n)*sum(G_1,2) - (1/n)*D.*c;

end