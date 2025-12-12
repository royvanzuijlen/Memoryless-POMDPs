clear all; close all; clc; warning off;
rng(0,'twister');
addpath('functions')

%% General settings
% Midsize example
nx = 40;  
ny = 20;  
nu = 10;
T = 20;

n_iter = 400;

MBPI.toggle = 1;
PG.toggle = 1;
ES.toggle = 0; % Infeasible for midsize example

%% Initialize system
% build ergodic transition kernels P(i,j,u) = Prob[x_{k+1}=j | x_k=i, u]
epsilon = 1e-9;
P = zeros(nx, nx, nu);
for u = 1:nu
    % create random positive matrix then normalize each row to sum to 1
    A = rand(nx, nx) + epsilon;
    A = A + 0.01*eye(nx);
    P(:,:,u) = bsxfun(@rdivide, A, sum(A,2));  % rows sum to 1
end

% build observation matrix O(y,i) = Prob[y | x=i]
% columns correspond to states; for each state i, O(:,i) is a distribution over observations
O = zeros(ny, nx);
for i = 1:nx
    v = rand(ny,1) + 1e-9;   % strictly positive
    O(:,i) = v / sum(v);
end

% stage-cost g(t,i,u) (we'll use rewards; higher is better)
% random costs in [0,1)
g = rand(T,nx, nu);

% Terminal value
Vt = rand(nx,1);

% Initial policy
pi_0 = ones(T,ny);

% Initial state distribution
mu_0 = rand(nx,1);
mu_0 = mu_0./sum(mu_0);

%% Model-based Policy Iteration

policy = cell(n_iter+1,1);
policy{1} = pi_0;
Q_xu = cell(n_iter+1,1);
Q_yu = cell(n_iter,1);
mu_all = cell(n_iter,1);
alpha_all = cell(n_iter,1);
J_test = zeros(n_iter,1);
J_nu = zeros(n_iter,T);

if MBPI.toggle==1
improvement_stage = 0;
switch_flag = 1; % 1 = forward, 2 = backward
for iter = 1:n_iter

    % Keep alternating between forward and backward sweeps
    if improvement_stage < T && switch_flag == 1 % Forward
        improvement_stage = improvement_stage + 1;
    elseif improvement_stage == T && switch_flag == 1 % Switch forward to backward
        improvement_stage = improvement_stage - 1;
        switch_flag = 2;
    elseif improvement_stage > 1 && switch_flag == 2 % Backward
        improvement_stage = improvement_stage - 1;
    else % Switch backward to forward
        improvement_stage = improvement_stage + 1;
        switch_flag = 1;
    end

    % Policy evaluation
    Q_xu{iter} = zeros(T,nx,nu); % h-1 +1
    
    % Last stage (T-1)
    for i=1:nx
        for u=1:nu
            Q_xu{iter}(T,i,u) = g(T,i,u);
            for j=1:nx
                Q_xu{iter}(T,i,u) = Q_xu{iter}(T,i,u) + P(i,j,u)*Vt(j);
            end
        end
    end
    

    % k=0:T-2
    for k=(T-1):-1:1
        Qnext = zeros(nx, ny);
        for o = 1:ny
            next_u = policy{iter}(k+1,o);
            Qnext(:,o) = Q_xu{iter}(k+1,:,next_u)';  % nx×1
        end
        expected_next = zeros(nx,1);
        for j = 1:nx
            expected_next(j) = sum(O(:,j) .* Qnext(j,:)');
        end
        % Then for each action u:
        for u = 1:nu
            Q_xu{iter}(k,:,u) = squeeze(g(k,:,u)) + (P(:,:,u) * expected_next)';
        end
    end

    % Policy improvement
    barmu = mu_0;
    Q_yu{iter} = zeros(T,ny,nu);
    mu_all{iter} = zeros(T+1,nx);
    mu_all{iter}(1,:) = barmu;
    alpha_all{iter} = zeros(T,ny,nx);
    for k=1:T
        gamma = O * mu_all{iter}(k,:)';            % ny×1
        alpha = (O .* mu_all{iter}(k,:))' ./ (gamma'); % nx×ny
        for j = 1:ny
            alpha_bar = alpha(:,j);
            barQ_nu = squeeze(alpha_bar' * reshape(Q_xu{iter}(k,:,:), nx, nu));
            alpha_all{iter}(k,j,:) = alpha_bar;
            if k==improvement_stage
                [~,policy{iter+1}(k,j)] = max(barQ_nu);
            else
                policy{iter+1}(k,j) = policy{iter}(k,j);
            end
            Q_yu{iter}(k,j,:) = barQ_nu;
        end

        % Update stationary distribution
        pi_k = policy{iter+1}(k,:);
        Pk = induced_P(P,O,pi_k');
        barmu = Pk'*barmu;
        mu_all{iter}(k+1,:) = barmu;
    end

end


nu_weight = zeros(ny,1);
for iter = 1:n_iter
  for k = 1:T
     nu_weight = (O * mu_all{iter}(k,:)')';  % 1×ny
     act_idx = policy{iter+1}(k,:);
     qvals = Q_yu{iter}(k,sub2ind(size(Q_yu{iter},[2 3]),1:ny,act_idx));
     J_nu(iter,k) = sum(nu_weight .* qvals);
  end
end

end
%% Policy gradient

PG.z0 = zeros(ny*nu*T,1);
PG.n_iter = 20;
PG.zhist = cell(PG.n_iter+1,1);
PG.zhist{1} = PG.z0;
PG.Jhist = zeros(PG.n_iter,1);
PG.alpha0 = 1;
PG.beta = 0.5;
PG.c = 1e-6;

if PG.toggle==1

for iter=1:PG.n_iter
    PG.Jhist(iter) = full_horizon_cost(P, O, g, Vt, mu_0, PG.zhist{iter}, ny, nu, T);
    
    % Analytic gradient (SLOW!!)
    % grad = full_horizon_grad(P, O, g, Vt, mu_0, PG.zhist{iter}, ny, nu, T);
        
    % Numerical gradient
    % grad = grad_est_fnc(P, O, g, Vt, mu_0, PG.zhist{iter}, ny, nu, T);

    % Matlab automatic differentiation (fastest)
    z0 = dlarray(PG.zhist{iter}');
    fun = @(z) full_horizon_cost(P, O, g, Vt, mu_0, z, ny, nu, T);
    [y, grad2] = dlfeval(@(z) modelGrad(z, fun), z0);
    grad = extractdata(grad2)';

    % Armijo backtracking
    PG.alpha = PG.alpha0;
    fx = PG.Jhist(iter);
    x0 = PG.zhist{iter};
    p = grad/max(abs(grad));
    fx_new = full_horizon_cost(P, O, g, Vt, mu_0, x0 + PG.alpha*p, ny, nu, T);
    while fx_new <= fx + PG.c*PG.alpha*grad'*p
        PG.alpha = PG.alpha*PG.beta;
        fx_new = full_horizon_cost(P, O, g, Vt, mu_0, x0 + PG.alpha*p, ny, nu, T);
        
    end
    PG.zhist{iter+1} = PG.zhist{iter} + PG.alpha*p;

end

end
%% Exhaustive search solution
ES.n_policies = nu^(ny*T);
ES.J_best = -inf;
if ES.toggle == 1 

for p=0:ES.n_policies-1
    policy_vec = dec2base(p,nu,ny*T) - '0' + 1;
    policy_mat = reshape(policy_vec,[T,ny]);
    J_current = evaluate_policy_td(P, O, g, Vt, policy_mat', mu_0);
    if J_current > ES.J_best
        ES.J_best = J_current;
        ES.best_policy = policy_mat;
    end
end

end

%% Figures
figure('name','Monotonic Weighted J convergence (1st stage)')
plot(1:n_iter,ones(1,n_iter)*ES.J_best,'k','linewidth',2)
hold on
plot(J_nu(:,1),'color',[0.9 0.6 0],'linewidth',2)
plot((1:PG.n_iter)*T,PG.Jhist,'color',[0.35 0.7 0.9],'linewidth',2)
grid on
xlabel('Number of stage updates','interpreter','latex')
ylabel('$L^{\pi}$','Interpreter','latex')
legend({'Exhaustive search','Model-based Policy iteration','Policy gradient'},'location','southeast')
xlim([0,n_iter]); xticks(0:50:n_iter);
ylim([10.5 12.5])



