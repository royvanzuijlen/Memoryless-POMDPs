function policy_det = stochastic_to_deterministic(policy_sto)
% STOCHASTIC_TO_DETERMINISTIC
% Convert a memoryless finite-horizon stochastic policy
%     policy_sto(t,y,a)
% into a deterministic one:
%     policy_det(t,y) ∈ {1,...,nu}
%
% Inputs:
%   policy_sto : (T+1) × ny × nu  stochastic policy
%
% Output:
%   policy_det : (T+1) × ny       deterministic actions

[T1, ny, nu] = size(policy_sto);

policy_det = zeros(T1, ny);

for t = 1:T1
    for y = 1:ny
        [~, best_a] = max(policy_sto(t, y, :));
        policy_det(t, y) = best_a;
    end
end

end