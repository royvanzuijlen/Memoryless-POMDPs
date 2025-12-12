function J_out = full_horizon_cost(P, O, g, gT, pi0, z, ny, nu, T)
% FULL_HORIZON_COST (optimized, identical results)

pi0 = pi0(:);
[nx,~,nu2] = size(P);
assert(nu2 == nu);

% ---- Precompute P' for each u, saves a transpose every iteration
PT = cell(nu,1);
for u = 1:nu
    PT{u} = P(:,:,u).';
end

% ---- Unpack logits once
Z = cell(T,1);
for t = 1:T
    block = z((t-1)*ny*nu+1 : t*ny*nu);
    Z{t} = reshape(block,[nu,ny]).'; % ny × nu
end

J = 0;
mu = pi0;

% Preallocate temp arrays
mu_next = zeros(nx,1);

for t = 1:T

    % ----------------------------------------------------
    % Softmax for each row y  (ny x nu)
    % ----------------------------------------------------
    E = exp(Z{t});
    Pi = E ./ sum(E,2);

    % ----------------------------------------------------
    % gamma(y) = P(y) = ∑_x μ(x) O(y|x)
    % ----------------------------------------------------
    gamma = O * mu;   % ny x 1

    % ----------------------------------------------------
    % Stage costs
    % ----------------------------------------------------
    g_t = squeeze(g(t,:,:));  % nx x nu

    mu_next(:) = 0;

    % ----------------------------------------------------
    % Loop over observations y, fully vectorize over u
    % ----------------------------------------------------
    for y = 1:ny
        gy = gamma(y);
        if gy == 0
            continue
        end

        % posterior α(x|y)
        alpha = (mu .* O(y,:).') / gy;

        % ---- expected immediate cost ----
        % alpha' g_t = 1 x nu
        ag = alpha.' * g_t;

        % J += gy * sum_u Pi(y,u) * ag(u)
        J = J + gy * sum(Pi(y,:) .* ag);

        % ---- next belief update ----
        % mu_next += gy * sum_u Pi(y,u) * (PT{u}*alpha)
        py = Pi(y,:);   % 1 x nu
        for u = 1:nu
            mu_next = mu_next + gy * py(u) * (PT{u} * alpha);
        end
    end

    % normalize belief
    mu = mu_next / sum(mu_next);
end

% terminal contribution
J = J + mu' * gT(:);
J_out = J;
end