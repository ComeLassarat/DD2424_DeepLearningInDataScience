function g = test_BatchNormBackPass2(G ,S, mu, v,epsilon)

% BatchNormBackProp
nb = size(G,2);

s1 = (v+eps).^(-0.5);
s2 = (v+eps).^(-1.5);
G1 = G.*(s1*ones(1,nb));
G2 = G.*(s2*ones(1,nb));
D = S-mu*ones(1,nb);
c = (G2.*D)*ones(nb,1);
g = G1 - (G1*ones(nb,1))*ones(1,nb)/nb - D.*(c*ones(1,nb))/nb;