clear all; close all; clc; warning off;
rng(0,'twister'); % Fixed random seed
addpath('functions')
savedata = 0;

%% Model definition
T = 5;   % Horizon - Fixed
nx = 20; % Number of states - Fixed

n_obs_act_vec = 2; % number of actions and observations (same number)
% n_obs_act_vec = 2:1:20; % number of actions and observations (same number)
n_sys = length(n_obs_act_vec);

%% Define the maximum number of actions/observations per method
% Model-based Policy Iteration
MBPI.max_n_obs_act = n_obs_act_vec(end);
% Model-based Policy gradient
PG.max_n_obs_act = n_obs_act_vec(end);
% Exhaustive search
ES.max_n_obs_act = 3;
% Mixed-integer linear programming (Cohen and Parmentier (2023))
MILP.max_n_obs_act = 3;
% Geometric method (Muller and Montufar (2022))
GEO.max_n_obs_act = 16;

%% Initialize systems
% Use time-invariant system matrices
P = cell(n_sys,1);
O = cell(n_sys,1);
g = cell(n_sys,1);
Vt = cell(n_sys,1);
pi_0 = cell(n_sys,1);
mu_0 = cell(n_sys,1);

for sys=1:n_sys
    nu = n_obs_act_vec(sys);
    ny = n_obs_act_vec(sys);

    % build ergodic transition kernels P(i,j,u) = Prob[x_{k+1}=j | x_k=i, u]
    epsilon = 1e-9;
    P{sys} = zeros(nx, nx, nu);
    for u = 1:nu
        % create random positive matrix then normalize each row to sum to 1
        A = rand(nx, nx) + epsilon;
        A = A + 0.01*eye(nx);
        P{sys}(:,:,u) = bsxfun(@rdivide, A, sum(A,2));  % rows sum to 1
    end

    % build observation matrix O(y,i) = Prob[y | x=i]
    % columns correspond to states; for each state i, O(:,i) is a distribution over observations
    O{sys} = zeros(ny, nx);
    for i = 1:nx
        v = rand(ny,1) + 1e-9;   % strictly positive
        O{sys}(:,i) = v / sum(v);
    end

    % stage-cost g(t,i,u) (we'll use rewards; higher is better)
    % random costs in [0,1)
    g{sys} = rand(T,nx, nu);

    % Terminal cost
    Vt{sys} = rand(nx,1);

    % Initial policy (always pick action 1)
    pi_0{sys} = ones(T,ny);

    % Initial state distribution
    mu_0{sys} = rand(nx,1);
    mu_0{sys} = mu_0{sys}./sum(mu_0{sys});
end

%% Policy Iteration
MBPI.n_iter = 1000; % Large number to store history
MBPI.J_final = zeros(n_sys,1);
MBPI.time = zeros(n_sys,1);
MBPI.max_iter = zeros(n_sys,1);
for sys = 1:n_sys
    tic
    nu = n_obs_act_vec(sys);
    ny = n_obs_act_vec(sys);
    policy = pi_0{sys};
    save_policy = zeros(size(policy));

    mu_all = zeros(T+1,nx);
    mu_all(1,:) = mu_0{sys};

    Q_yu = cell(MBPI.n_iter,1);
    alpha_all = cell(MBPI.n_iter,1);
    J_test = zeros(MBPI.n_iter,1);


    % Initialize Q
    % Policy evaluation
    Q_xu = zeros(T,nx,nu); % h-1 +1
    
    % Last stage (T-1)
    for i=1:nx
        for u=1:nu
            Q_xu(T,i,u) = g{sys}(T,i,u);
            for j=1:nx
                Q_xu(T,i,u) = Q_xu(T,i,u) + P{sys}(i,j,u)*Vt{sys}(j);
            end
        end
    end
    

    % k=0:T-2
    for k=(T-1):-1:1
        Qnext = zeros(nx, ny);
        for o = 1:ny
            next_u = policy(k+1,o);
            Qnext(:,o) = Q_xu(k+1,:,next_u)';  % nx×1
        end
        expected_next = zeros(nx,1);
        for j = 1:nx
            expected_next(j) = sum(O{sys}(:,j) .* Qnext(j,:)');
        end
        % Then for each action u:
        for u = 1:nu
            Q_xu(k,:,u) = squeeze(g{sys}(k,:,u)) + (P{sys}(:,:,u) * expected_next)';
        end
    end
    
    % First improvement
    gamma = O{sys} * mu_all(1,:)';            % ny×1
    alpha = (O{sys} .* mu_all(1,:))' ./ (gamma'); % nx×ny
    for j = 1:ny
        alpha_bar = alpha(:,j);
        barQ_nu = squeeze(alpha_bar' * reshape(Q_xu(1,:,:), nx, nu));
        [~,policy(1,j)] = max(barQ_nu);
    end

    % Main sweeps
    t = 1;
    switch_flag = 1; % 1 = forward, 2 = backward

    for iter = 1:MBPI.n_iter
        % Keep alternating between forward and backward sweeps
        if t < T && switch_flag == 1 % Forward
            
            % Update stationary distribution
            pi_t = policy(t,:);
            Pk = induced_P(P{sys},O{sys},pi_t');
            mu_all(t+1,:) = Pk'*mu_all(t,:)';

            t = t + 1;
        elseif t == T && switch_flag == 1 % Switch forward to backward
            % Update Q table
            Qnext = zeros(nx, ny);
            for o = 1:ny
                next_u = policy(t,o);
                Qnext(:,o) = Q_xu(t,:,next_u)';  % nx×1
            end
            expected_next = zeros(nx,1);
            for j = 1:nx
                expected_next(j) = sum(O{sys}(:,j) .* Qnext(j,:)');
            end
            % Then for each action u:
            for u = 1:nu
                Q_xu(t-1,:,u) = squeeze(g{sys}(t-1,:,u)) + (P{sys}(:,:,u) * expected_next)';
            end

            t = t - 1;
            switch_flag = 2;
        elseif t > 1 && switch_flag == 2 % Backward

            % Update Q table
            Qnext = zeros(nx, ny);
            for o = 1:ny
                next_u = policy(t,o);
                Qnext(:,o) = Q_xu(t,:,next_u)';  % nx×1
            end
            expected_next = zeros(nx,1);
            for j = 1:nx
                expected_next(j) = sum(O{sys}(:,j) .* Qnext(j,:)');
            end
            % Then for each action u:
            for u = 1:nu
                Q_xu(t-1,:,u) = squeeze(g{sys}(t-1,:,u)) + (P{sys}(:,:,u) * expected_next)';
            end

            t = t - 1;
        else % Switch backward to forward
            save_policy = policy;
            % Update stationary distribution
            pi_t = policy(t,:);
            Pk = induced_P(P{sys},O{sys},pi_t');
            mu_all(t+1,:) = Pk'*mu_all(t,:)';

            t = t + 1;
            switch_flag = 1;
        end

        % Policy improvement
        gamma = O{sys} * mu_all(t,:)';            % ny×1
        alpha = (O{sys} .* mu_all(t,:))' ./ (gamma'); % nx×ny
        for j = 1:ny
            alpha_bar = alpha(:,j);
            barQ_nu = squeeze(alpha_bar' * reshape(Q_xu(t,:,:), nx, nu));
            [~,policy(t,j)] = max(barQ_nu);
        end

        if all(all(policy == save_policy)) && t==1
            Qnext = zeros(nx, ny);
            for o = 1:ny
                next_u = policy(t,o);
                Qnext(:,o) = Q_xu(t,:,next_u)';  % nx×1
            end
            expected_next = zeros(nx,1);
            for j = 1:nx
                expected_next(j) = sum(O{sys}(:,j) .* Qnext(j,:)');
            end
            MBPI.J_final(sys) = expected_next'*mu_0{sys};
            MBPI.time(sys) = toc;
            MBPI.max_iter(sys) = iter;
            break;
            
        end
    end

end

%% Policy gradient
PG.J_final = zeros(n_sys,1);
PG.time = zeros(n_sys,1);
PG.max_iter = zeros(n_sys,1);
PG.delta = 1e-6;

for sys=1:n_sys
    tic
    nu = n_obs_act_vec(sys);
    ny = n_obs_act_vec(sys);

    if nu > PG.max_n_obs_act
        break;
    end

    PG.z0 = zeros(ny*nu*T,1);
    PG.n_iter = 500000; % large number to store history
    PG.zhist = cell(PG.n_iter+1,1);
    PG.zhist{1} = PG.z0;
    PG.Jhist = zeros(PG.n_iter,1);
    PG.alpha0 = 1;
    PG.beta = 0.5;
    PG.c = 1e-6;
    
    diff = 1;
    iter = 0;
    

    while diff > PG.delta
        if iter <= 1
            diff = 1;
        else
            diff = abs(PG.Jhist(iter) - PG.Jhist(iter-1));
        end
        iter = iter+1;
        
        PG.Jhist(iter) = full_horizon_cost(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, PG.zhist{iter}, ny, nu, T);

        % Analytic gradient (SLOW!!)
        % grad = full_horizon_grad(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, PG.zhist{iter}, ny, nu, T);
        
        % Numerical gradient
        % grad = grad_est_fnc(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, PG.zhist{iter}, ny, nu, T);

        % Matlab automatic differentiation (fastest)
        z0 = dlarray(PG.zhist{iter}');
        fun = @(z) full_horizon_cost(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, z, ny, nu, T);
        [y, grad2] = dlfeval(@(z) modelGrad(z, fun), z0);
        grad = extractdata(grad2)';

        % Armijo backtracking
        PG.alpha = PG.alpha0;
        fx = PG.Jhist(iter);
        x0 = PG.zhist{iter};
        p = grad/max(abs(grad));

        fx_new = full_horizon_cost(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, x0 + PG.alpha*p, ny, nu, T);
        while fx_new < fx + PG.c*PG.alpha*(grad'*p)
            PG.alpha = PG.alpha*PG.beta;
            fx_new = full_horizon_cost(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, x0 + PG.alpha*p, ny, nu, T);
        end
        PG.zhist{iter+1} = x0 + PG.alpha*p;
    end

    PG.J_final(sys) = PG.Jhist(iter);
    PG.time(sys) = toc;
    PG.max_iter(sys) = iter;
end
%% Exhaustive search solution
ES.J_final = zeros(n_sys,1);
ES.time = zeros(n_sys,1);

for sys=1:n_sys
    tic
    nu = n_obs_act_vec(sys);
    ny = n_obs_act_vec(sys);

    if nu > ES.max_n_obs_act
        break;
    end

    ES.n_policies = nu^(ny*T);
    ES.J_best = -inf;
    
    for p=0:ES.n_policies-1
        policy_vec = dec2base(p,nu,ny*T) - '0' + 1;
        policy_mat = reshape(policy_vec,[T,ny]);
        J_current = evaluate_policy_td(P{sys}, O{sys}, g{sys}, Vt{sys}, policy_mat, mu_0{sys});
        if J_current > ES.J_best
            ES.J_best = J_current;
            ES.best_policy = policy_mat;
        end
    end
        
    ES.J_final(sys) = ES.J_best;
    ES.time(sys) = toc;

end
%% Mixed integer linear programming (Cohen '23)
opts.solver = 'gurobi'; % or 'intlinprog' if you don't have gurobi

MILP.J_final = zeros(n_sys,1);
MILP.time = zeros(n_sys,1);

for sys=1:n_sys
    tic
    nu = n_obs_act_vec(sys);
    ny = n_obs_act_vec(sys);

    if nu > MILP.max_n_obs_act
        break;
    end

    [MILP.policy_MILP{sys}, ~,solinfo] = memoryless_policy_milp(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, T-1, opts);
    MILP.time(sys) = toc;
    J1 = evaluate_policy_td(P{sys}, O{sys}, g{sys}, Vt{sys}, MILP.policy_MILP{sys}, mu_0{sys});
    MILP.J_final(sys) = J1;

end


%% Geometry approach (Muller '22)
opts.solver = 'fmincon';

GEO.J_final = zeros(n_sys,1);
GEO.time = zeros(n_sys,1);

for sys=1:n_sys
    tic
    nu = n_obs_act_vec(sys);
    ny = n_obs_act_vec(sys);
    
    if nu > GEO.max_n_obs_act
        break;
    end

    [GEO.policy_GEO_stoch{sys}, ~,solinfo] = memoryless_policy_geo(P{sys}, O{sys}, g{sys}, Vt{sys}, mu_0{sys}, T-1, opts);
    GEO.time(sys) = toc;
    GEO.policy_GEO_det{sys} = stochastic_to_deterministic(GEO.policy_GEO_stoch{sys});
    J1 = evaluate_policy_td(P{sys}, O{sys}, g{sys}, Vt{sys}, GEO.policy_GEO_det{sys}, mu_0{sys});
    GEO.J_final(sys) = J1;
end

%% Save workspace
if savedata == 1
% Get current date and time
t = datetime('now');

% Format as yyyy_mm_dd_hh_mi
filename = sprintf('Comp_Time_data_%04d_%02d_%02d_%02d_%02d.mat', year(t), month(t), day(t), hour(t), minute(t));

% Save the entire workspace
save(filename);
end

%% Figures
figure('name','Computation time')
semilogy(n_obs_act_vec,ES.time,'k','linewidth',2)
hold on; grid on
semilogy(n_obs_act_vec,MBPI.time,'color',[0.9 0.6 0],'linewidth',2)
semilogy(n_obs_act_vec,PG.time,'color',[0.35 0.7 0.9],'linewidth',2)
semilogy(n_obs_act_vec,MILP.time,'color',[0 0.6 0.5],'linewidth',2)
semilogy(n_obs_act_vec,GEO.time,'color',[0.8 0.4 0],'linewidth',2)
xticks(n_obs_act_vec)
xlabel('$|\mathcal{O}|$ and $|\mathcal{A}|$','interpreter','latex')
ylabel('Computation time [s]','interpreter','latex')
legend('Exhaustive Search','Model-based Policy Iteration','Policy gradient','Mixed-integer linear programming','Geometric')