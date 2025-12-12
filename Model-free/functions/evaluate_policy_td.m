function J = evaluate_policy_td(P, O, g, gT, mu_seq, pi0)
% EVALUATE_POLICY_TD  Evaluate deterministic time-varying policy with time-dependent and terminal costs
%
%   J = evaluate_policy_td(P, O, g, gT, mu_seq, pi0)
%
% Inputs:
%   P      : nx x nx x nu      transition probabilities (P(i,j,u))
%   O      : ny x nx           observation probabilities (O(y,i))
%   g      : T x nx x nu       stage costs (time-dependent)
%   gT     : nx x 1            terminal cost vector
%   mu_seq : ny x T            policy sequence; column t = mu_t(y) ∈ {1..nu}
%   pi0    : nx x 1            initial belief
%
% Output:
%   J : scalar expected total cost (sum of stage costs + terminal cost)
%
% Notes:
%   - Observation-before-action convention.
%   - Efficient for brute-force evaluation loops.
%

mu_seq = transpose(mu_seq);
[nx,~,nu] = size(P);
[ny, nxO] = size(O);
assert(nx == nxO, 'O must be ny x nx');
T = size(mu_seq,2);
assert(size(g,1) == T && size(g,2) == nx && size(g,3) == nu, ...
    'g must be T x nx x nu');

b = pi0(:) / sum(pi0(:));  % belief
J = 0;                     % total expected cost

for t = 1:T
    mu_t = mu_seq(:,t);
    g_t = squeeze(g(t,:,:));   % nx x nu
    gamma = O * b;             % observation probabilities

    c_t = 0;
    next_b_weighted = zeros(nx,1);

    for y = 1:ny
        a = mu_t(y);
        if gamma(y) > 0
            alpha = (b .* O(y,:)') / gamma(y);
            c_t = c_t + gamma(y) * (alpha' * g_t(:,a));
            b_next = P(:,:,a)' * alpha;
            next_b_weighted = next_b_weighted + gamma(y) * b_next;
        end
    end

    J = J + c_t;
    b = next_b_weighted / sum(next_b_weighted);
end

% add terminal cost
J = J + b' * gT(:);

end