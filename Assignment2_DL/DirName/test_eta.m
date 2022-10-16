eta = zeros(3465,1);
x = linspace(0,3465,3466);

for i=0:3465
    eta(i+1) = ComputeEta(1e-5,1e-2,800,i);
end

plot(x,eta)