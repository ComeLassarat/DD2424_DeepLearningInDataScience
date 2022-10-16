function eta = ComputeEta(eta_min, eta_max, n_s, iteration)

t = mod(iteration,2*n_s);

if (0 <= t) && (t <= n_s)
    eta = eta_min + (t/n_s)*(eta_max-eta_min);
end

if (n_s < t) && (t <= 2*n_s)
    eta = eta_max - ((t-n_s)/n_s)*(eta_max-eta_min);
end


end