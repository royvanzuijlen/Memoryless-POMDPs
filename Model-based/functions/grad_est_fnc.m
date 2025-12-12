function [grad] = grad_est_fnc(P, O, g, gT, pi0, z, ny, nu, T)
    n_z = length(z);
    grad = zeros(n_z,1);
    parfor i=1:n_z
        epsilon    = zeros(n_z,1); 
        epsilon(i) = 1e-8;
        grad(i) = (full_horizon_cost(P, O, g, gT, pi0, z+epsilon, ny, nu, T)-full_horizon_cost(P, O, g, gT, pi0, z, ny, nu, T))/(epsilon(i));
    end
end