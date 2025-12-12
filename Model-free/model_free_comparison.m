clear all; close all; clc; warning off;
addpath('functions')
rng(0,'twister');

%% General settings
nx = 5;  
ny = 5;  
nu = 5;
T = 10;

n_iter = 31;
n_MC = 5;
n_simulations = 5000;

%% System definition
state_vec = 1:nx;
obs_vec = 1:ny;
action_vec = 1:nu;

% build ergodic transition kernels P(i,j,u) = Prob[x_{k+1}=j | x_k=i, u]
% Guarantee irreducible & aperiodic by making every entry positive (small epsilon)
epsilon = 1e-9;
P = zeros(nx, nx, nu);
for u = 1:nu
    % create random positive matrix then normalize each row to sum to 1
    A = rand(nx, nx) + epsilon;
    % add a bit of weight to diagonal to ensure aperiodicity
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

% stage-cost g(i,u) (we'll use rewards; higher is better)
% random costs in [0,1)
g = rand(T,nx, nu);

% Terminal cost
Vt = rand(nx,1);

% Initial policy
pi_0 = ones(T,ny);

% Initial state distribution
mu_0 = rand(1,nx);
mu_0 = mu_0./sum(mu_0);


%% LSPI
Q_LS.n_iter = n_iter;
Q_LS.n_simulations = n_simulations;
Q_LS.J_test = zeros(n_MC,Q_LS.n_iter);
Q_LS.policy = cell(Q_LS.n_iter,1);

epsilon = 0.1;            % behavior policy ε-greedy during data collection
reg_lambda = 1e-8;        % regularization for A matrix
policy_eval_iters = 1;    % number of LSTDQ evaluation passes per outer iter
gamma = 1.0;              % for finite horizon set gamma = 1. For discounted, set <1

% ---------- Feature mapping ----------
% Default: one-hot per (t,o,a) (dimension = T * ny * nu).
d = T * ny * nu;
phi_func = @(t,o,a) one_hot_phi(t,o,a,T,ny,nu);  % returns column vector (d x 1)
% ------------------------------------------------

% initialize policy randomly (for first data collection)
policy_cur = randi(nu, T, ny);

if 1==1
for MC=1:n_MC
for iter = 1:Q_LS.n_iter
    % --- Collect batch of transitions under epsilon-greedy using policy_cur ---
    % DATA: [t, obs, action, reward, obsprime]
    % store as DATA(t,sim,:) in cell-like array (but we'll use 2D array)
    DATA = zeros(T, Q_LS.n_simulations, 5); % [t, o, a, r, o']
    for sim = 1:Q_LS.n_simulations
        x = randsrc(1,1,[state_vec; mu_0]);
        obs = randsrc(1,1,[obs_vec; O(:,x)']);
        for t = 1:T
            % behavior policy: epsilon-greedy on current policy_cur (or random if none)
            if rand() > epsilon
                a = policy_cur(t, obs);
            else
                a = randi(nu);
            end

            xprime = randsrc(1,1,[state_vec; P(x,:,a)]);
            obsprime = randsrc(1,1,[obs_vec; O(:,xprime)']);
            r = g(t, x, a);
            if t == T
                r = r + Vt(xprime);
            end

            DATA(t, sim, :) = [t, obs, a, r, obsprime];

            x = xprime;
            obs = obsprime;
        end
    end

    % Flatten dataset into list of transitions for easy accumulation
    % rows: N = T * n_simulations
    N = T * Q_LS.n_simulations;
    transitions = zeros(N,5); % [t, o, a, r, o']
    idx = 1;
    for t=1:T
        for s=1:Q_LS.n_simulations
            transitions(idx, :) = squeeze(DATA(t,s,:))';
            idx = idx + 1;
        end
    end

    % --- Policy evaluation by LSTDQ (finite-horizon) ---
    % We'll compute theta_t for each time step t (stacked or cell).
    Theta = zeros(d, T);  % each column theta(:,t) corresponds to Q_t approx

    % Start with the current policy_cur as the evaluation policy.
    % We'll perform `policy_eval_iters` LSTDQ sweeps (usually 1 suffices for batch LS).
    for pe = 1:policy_eval_iters
        % For each time step t, accumulate A_t and b_t using transitions that occurred at time t.
        for t = T:-1:1
            A = zeros(d,d);
            b = zeros(d,1);

            % iterate over transitions that happened at time t
            rows = find(transitions(:,1) == t);
            
            ny_nu = ny * nu;

            for rr = rows'
                o  = transitions(rr,2);
                a  = transitions(rr,3);
                r  = transitions(rr,4);
                op = transitions(rr,5);

                % current index
                i = (t-1)*ny_nu + (o-1)*nu + a;

                % A(i,i) += 1
                A(i,i) = A(i,i) + 1;

                if t < T
                    a_next = policy_cur(t+1, op);
                    j = t * ny_nu + (op-1)*nu + a_next;

                    % A(i,j) -= gamma
                    A(i,j) = A(i,j) - gamma;
                end

                % b(i) += r
                b(i) = b(i) + r;
            end

            % regularize and solve
            A = A + reg_lambda * eye(d);
            % numeric safety: if A near-singular, use pinv
            theta_t = A \ b;  % d x 1
            Theta(:, t) = theta_t;
        end
    end

    % --- Policy improvement (greedy w.r.t. estimated Q) ---
    policy_new = zeros(T, ny);
    for t = 1:T
        for o = 1:ny
            qvals = zeros(nu,1);
            for a = 1:nu
                phi = phi_func(t, o, a);
                qvals(a) = phi' * Theta(:, t);
            end
            [~, best_a] = max(qvals);
            policy_new(t,o) = best_a;
        end
    end

    policy_cur = policy_new;           % update current policy
    Q_LS.policy{iter} = policy_cur;

    % optional: compute J_test using your evaluation function (assumes policy uses obs index)
    Q_LS.J_test(MC,iter) = evaluate_policy_td(P, O, g, Vt, policy_cur, mu_0);

    fprintf('LSPI iter %d done. J_test = %.6g\n', iter, Q_LS.J_test(MC,iter));
end
end
end
%% State-informed model-free policy iteration
SIMF.n_iter = n_iter;
SIMF.ns_DATA = n_simulations;                % number of episodes per dataset (state-informed)
SIMF.n_sweeps = 1;         % how many forward+backward sweeps per dataset
epsilon_behave =0.5;           % epsilon for behaviour pi -> b (epsilon-soft)
SIMF.policy = cell(SIMF.n_iter+1,1);
SIMF.policy{1} = pi_0;
SIMF.J_test = zeros(n_MC,SIMF.n_iter);

for MC=1:n_MC
for iter = 1:SIMF.n_iter
    DATA = zeros(T+1, SIMF.ns_DATA, 6);
    for sl = 1:SIMF.ns_DATA
         sIdx = randsrc(1,1,[state_vec;mu_0]);
         for t=1:T
            oIdx = randsrc(1,1,[obs_vec;O(:,sIdx)']);
            
            a_det = SIMF.policy{iter}(t,oIdx);
            b = ones(1,nu) * (epsilon_behave/nu);
            b(a_det) = b(a_det) + (1 - epsilon_behave);
            

            aIdx = randsrc(1,1,[1:nu; b]);
            b_prob = b(aIdx);
            reward = g(t,sIdx,aIdx);

            sprimeIdx = randsrc(1,1,[state_vec;P(sIdx,:,aIdx)]);
         
            % Save on-policy data
            DATA(t,sl,:) = [sIdx,oIdx,aIdx,reward,sprimeIdx,b_prob];
                
            sIdx = sprimeIdx; % Update state index
         end
         % Save terminal values
         reward = Vt(sIdx);
         DATA(T+1,sl,:) = [sIdx,0,0,reward,0,0];
    end

    % 2) Precompute convenient counts from DATA (Nt(s), Nt(s,o), Nt(o), Nt(s,a))
    % We'll index t=1..T (DATA rows 1..T) and terminal row T+1 used for VT estimates.
    % Precompute counts for all t
    Nt_s = zeros(T+1, nx);
    Nt_so = zeros(T+1, nx, ny);
    Nt_o = zeros(T+1, ny);
    Nt_sa = zeros(T+1, nx, nu);

    for t = 1:T+1
        for s = 1:nx
            Nt_s(t,s) = sum(squeeze(DATA(t,:,1)) == s);
            for o=1:ny
                Nt_so(t,s,o) = sum( (squeeze(DATA(t,:,1)) == s) & (squeeze(DATA(t,:,2)) == o) );
            end
            for a=1:nu
                Nt_sa(t,s,a) = sum( (squeeze(DATA(t,:,1)) == s) & (squeeze(DATA(t,:,3)) == a) );
            end
        end
        for o=1:ny
            Nt_o(t,o) = sum(squeeze(DATA(t,:,2)) == o);
        end
    end

    % Estimate terminal value
    Vhat_T = zeros(nx,1);
    for s=1:nx
        Reward_sum = sum((DATA(T+1,:,1)==s).*DATA(T+1,:,4));
        Vhat_T(s) = Reward_sum/Nt_s(T+1,s);
        if isnan(Vhat_T(s))
            Vhat_T(s) = -1e10;
        end
    end

    % Estimate observation model
    O_hat = zeros(ny,nx);
    for s=1:nx
        for o = 1:ny
            O_hat(o,s) = sum(squeeze(Nt_so(1:T,s,o)))/sum(Nt_s(1:T,s));
        end
    end

    % Policy iteration
    Q_sa = zeros(T,nx,nu);
    for t=T:-1:1
        % Estimate state-based Q-function of initial policy
        for s = 1:nx
            for a = 1:nu
                % Nt_sa = sum(DATA(t,:,1)==s & DATA(t,:,3)==a);
                if t==T
                    dummy = (DATA(t,:,1)==s & DATA(t,:,3)==a).*(DATA(t,:,4) + Vhat_T(DATA(t,:,5))');
                    dummy(isnan(dummy)) = 0;
                    Reward_sum = sum(dummy);
                else
                    dummy = (DATA(t,:,1)==s & DATA(t,:,3)==a).*(DATA(t,:,4) + Vnext(DATA(t,:,5))');
                    dummy(isnan(dummy)) = 0;
                    Reward_sum = sum(dummy);
                end
                Q_sa(t,s,a) = Reward_sum/Nt_sa(t,s,a);
            end
        end


        % Compute new values
        Vnext = zeros(nx,1);
        for s=1:nx
            for o=1:ny
                Vnext(s) = Vnext(s) + O_hat(o,s)*squeeze(Q_sa(t,s,SIMF.policy{iter}(t,o)));
            end
        end

    end

    % 7) Sweeps: perform NM.n_sweeps forward+backward passes on the single DATA
    policy_cur = SIMF.policy{iter};
    alpha_hat = cell(T,1);
    for sweep = 1:SIMF.n_sweeps
        
        % ---- Forward sweep (t = 1 .. T-1)  (maps to paper's t=0..T-2) ----
        for t = 1:(T-1)
            if t == 1
                Ntilde_prev = squeeze(Nt_so(1,:,:))'; % nx x ny -> careful orientation
            end

            Ntilde_t = Ntilde_prev; % shape nx x ny (s x o)
            alpha_hat{t} = Ntilde_t./sum(Ntilde_t,2);

            % Estimate alpha_hat at stage t using importance-corrected counters (Appendix B)
            % Build M_l(t): weighted counts of transitions (s,o) -> (s',o') with rho corrections
            M = zeros(nx, ny, nx, ny); % M(s,o; s',o')
            for ep = 1:SIMF.ns_DATA
                s = DATA(t,ep,1);
                o = DATA(t,ep,2);
                a = DATA(t,ep,3);
                s_next = DATA(t,ep,5);
                o_next = DATA(t+1,ep,2); % next observation sampling

                % importance ratio pi_t(a|o) / b_t(a|o)
                if policy_cur(t,o) == a
                    pi_action = 1;
                else
                    pi_action = 0;
                end
                b_prob = DATA(t,ep,6);
                if b_prob <= 0
                    rho = 0;
                else
                    rho = pi_action / b_prob;
                end
                M(s,o,s_next,o_next) = M(s,o,s_next,o_next) + rho;
            end

            % iterate recursion up to stage t
            Ntilde_next = zeros(nx, ny);
            for s=1:nx
                for o=1:ny
                    denom = Nt_so(t, s, o);
                    if denom==0, continue; end
                    for sp=1:nx
                        for op=1:ny
                            Ntilde_next(sp,op) = Ntilde_next(sp,op) + Ntilde_prev(s,o) * M(s,o,sp,op) / denom;
                        end
                    end
                end
            end
            Ntilde_prev = Ntilde_next';

            % Build Q_bar (o,a) as weighted sum by alpha_hat (eq 11)
            for o=1:ny
                alpha_bar = alpha_hat{t}(o,:);
                barQ_nu = squeeze(alpha_bar * reshape(Q_sa(t,:,:), nx, nu));
                [~, best_a] = max(barQ_nu);
                policy_cur(t,o) = best_a;
            end           
        end % forward t loop

        % Alpha of last improvement stage
        Ntilde_t = Ntilde_prev; % shape nx x ny (s x o)
        alpha_hat{T} = Ntilde_t./sum(Ntilde_t,2);

        % ---- Backward sweep (t = T .. 2)  (maps to paper's t=T-1 .. 1) ----
        for t = T:-1:2
            for s = 1:nx
                for a = 1:nu
                    % Nt_sa = sum(DATA(t,:,1)==s & DATA(t,:,3)==a);
                    if t==T
                        dummy = (DATA(t,:,1)==s & DATA(t,:,3)==a).*(DATA(t,:,4) + Vhat_T(DATA(t,:,5))');
                        dummy(isnan(dummy)) = 0;
                        Reward_sum = sum(dummy);
                    else
                        dummy = (DATA(t,:,1)==s & DATA(t,:,3)==a).*(DATA(t,:,4) + Vnext(DATA(t,:,5))');
                        dummy(isnan(dummy)) = 0;
                        Reward_sum = sum(dummy);
                    end
                    Q_sa(t,s,a) = Reward_sum/Nt_sa(t,s,a);
                end
            end
            
            % Compute new values
            Vnext = zeros(nx,1);
            for s=1:nx
                for o=1:ny
                    Vnext(s) = Vnext(s) + O_hat(o,s)*squeeze(Q_sa(t,s,policy_cur(t,o)));
                end
            end

            % Build Q_bar (o,a) as weighted sum by alpha_hat (eq 11)
            for o=1:ny
                alpha_bar = alpha_hat{t}(o,:);
                barQ_nu = squeeze(alpha_bar * reshape(Q_sa(t,:,:), nx, nu));
                [~, best_a] = max(barQ_nu);
                policy_cur(t,o) = best_a;
            end  

            % After improving at t in backward sweep, recompute Q_{t-1} would be needed.
            % Our next backward loop iteration (t-1) will compute Q_hat at t-1 with the updated policy_cur,
            % using the same DATA and the updated policy, so Q propagation occurs by re-evaluating Q_hat in each step.
        end % backward t loop

        % Update Q values of first stage
        for s = 1:nx
            for a = 1:nu
                dummy = (DATA(1,:,1)==s & DATA(1,:,3)==a).*(DATA(1,:,4) + Vnext(DATA(1,:,5))');
                dummy(isnan(dummy)) = 0;
                Reward_sum = sum(dummy);
            end
            Q_sa(1,s,a) = Reward_sum/Nt_sa(1,s,a);
        end

    end % sweeps

    % Save updated policy
    SIMF.policy{iter+1} = policy_cur;

    % Evaluate the current improved policy using your provided evaluator
    SIMF.J_test(MC,iter) = evaluate_policy_td(P, O, g, Vt, SIMF.policy{iter+1}, mu_0);
    fprintf('State-informed PI iter %d done. J_test = %.6g\n', iter, SIMF.J_test(MC,iter));

end % iter loop
end
%% Policy gradient
PG.J_test = zeros(n_MC,n_iter);
PG.n_features = ny*nu*T;
PG.Phi = eye(PG.n_features);
PG.theta = zeros(PG.n_features,1);
PG.alpha = 0.001;
PG.n_iter = n_iter;
PG.n_simulations = n_simulations;

if 1==1
for MC = 1:n_MC
for iter=1:PG.n_iter
    J_temp = zeros(PG.n_simulations,1);
    for sim=1:PG.n_simulations
        % Simulation
        Data = zeros(T,3);
        data_count = 1;
        x = randsrc(1,1,[state_vec;mu_0]);
        obs = randsrc(1,1,[obs_vec;O(:,x)']);
        for t=1:T
            policy_prob = sm_pol_prob(obs,t,nu,T,PG);
            action = randsrc(1,1,[action_vec;policy_prob']);

            xprime = randsrc(1,1,[state_vec;P(x,:,action)]);
            obsprime = randsrc(1,1,[obs_vec;O(:,xprime)']);
            reward = g(t,x,action);
            if t==T
                reward = reward + Vt(xprime); % Add terminal cost at last stage.
            end

            J_temp(sim) = J_temp(sim) + reward;

            % Save data
            Data(data_count,:) = [obs,action,reward];
            data_count = data_count+1;

            x = xprime;
            obs = obsprime;
        end

        % Learning
        G = 0;
        for k=T:-1:1
            St = Data(k,1);
            At = Data(k,2);
            Rtplus1 = Data(k,3);
            G = G + Rtplus1;
            feature_idx = state_to_idx(St,At,k,nu,T);
            feature_vec = PG.Phi(:,feature_idx);
        
            policy_prob = sm_pol_prob(St,k,nu,T,PG);
            temp = zeros(PG.n_features,1);
            for i=1:nu
                feature_idx_temp = state_to_idx(St,i,k,nu,T);
                feature_vec_temp = PG.Phi(:,feature_idx_temp);
                temp = temp + policy_prob(i)*feature_vec_temp;
            end
            policy_grad = feature_vec - temp;
            % policy_grad = policy_grad/max(abs(policy_grad));
            PG.theta = PG.theta + PG.alpha*G*policy_grad;
            % PG.theta = PG.theta + sqrt(0.01/iter)*G*policy_grad;
        end
    end

    PG.policy{iter} = zeros(T,ny);
    for k=1:T
        for i=1:ny
            policy_prob = sm_pol_prob(i,k,nu,T,PG);
           [~,PG.policy{iter}(k,i)] = max(policy_prob);
        end
    end
       
    PG.J_test(MC,iter) = evaluate_policy_td(P, O, g, Vt, PG.policy{iter}, mu_0);
    fprintf('Policy gradient iter %d done. J_test = %.6g\n', iter, PG.J_test(MC,iter));

end
end
end
%% Observation-only model-free policy iteration
OOMF.n_iter = n_iter;
OOMF.n_simulations = n_simulations;
OOMF.J_test = zeros(n_MC,OOMF.n_iter);
OOMF.policy = cell(OOMF.n_iter,1);

if 1==1
for MC=1:n_MC
OOMF.table = zeros(T+1,ny,nu);

imp_stage = 0;
for iter=1:OOMF.n_iter
    DATA_Q = zeros(T,OOMF.n_simulations,4);
    imp_stage = imp_stage + 1;
    if imp_stage > T
        imp_stage = 1;
    end
    for sim=1:OOMF.n_simulations
        x = randsrc(1,1,[state_vec;mu_0]);
        obs = randsrc(1,1,[obs_vec;O(:,x)']);
        for t=1:T
            if t == imp_stage
                action = randi(nu);
            else
                [~,action] = max(OOMF.table(t,obs,:));
            end
            xprime = randsrc(1,1,[state_vec;P(x,:,action)]);
            obsprime = randsrc(1,1,[obs_vec;O(:,xprime)']);
            reward = g(t,x,action);
            if t==T
                reward = reward + Vt(xprime); % Add terminal cost at last stage.
            end
            
            DATA_Q(t,sim,:) = [obs,action,reward,obsprime];

            x = xprime;
            obs = obsprime;
        end
    end

    % Initialize Q
    OOMF.table = zeros(T+1,ny,nu);

    % returns for each sample at next time step
    Gnext = [];   % will be overwritten per episode

    % loop backward in time
    for t = T:-1:1
        data = squeeze(DATA_Q(t,:,:));  % [o, a, r, o']

        o  = data(:,1);
        a  = data(:,2);
        r  = data(:,3);

        if t == T
            G = r;                            % terminal return
        else
            G = r + Gnext;           % MC return
        end

        % average MC returns per (o,a)
        OOMF.table(t,:,:) = accumarray([o a],G,[ny nu],@mean,0 );

        Gnext = G;    % store for previous step
    end

    OOMF.policy{iter} = zeros(T,ny);
    for t=1:T
        for y=1:ny
            [~,OOMF.policy{iter}(t,y)] = max(OOMF.table(t,y,:));
        end
    end

    OOMF.J_test(MC,iter) = evaluate_policy_td(P, O, g, Vt, OOMF.policy{iter}, mu_0);
    fprintf('Observation-only PI iter %d done. J_test = %.6g\n', iter, OOMF.J_test(MC,iter));
end
end
end

%% Figures
figure(1)
plot(SIMF.J_test','k','linewidth',2)
hold on; grid on
plot(OOMF.J_test','color',[0.9 0.6 0],'linewidth',2)
plot(PG.J_test','color',[0.35 0.7 0.9],'linewidth',2)
plot(Q_LS.J_test','color',[0 0.6 0.5],'linewidth',2)
ylim([6.8 7.3])
xlim([1 n_iter])
xticks([1:5:n_iter])
legend('State-informed model-free','Observation-only model-free','Policy gradient','LSPI','location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('$L^{\pi}$','Interpreter','latex')

figure(3)
plot(mean(SIMF.J_test,1),'k','linewidth',2)
hold on; grid on
plot(mean(OOMF.J_test,1),'color',[0.9 0.6 0],'linewidth',2)
plot(mean(PG.J_test,1),'color',[0.35 0.7 0.9],'linewidth',2)
plot(mean(Q_LS.J_test,1),'color',[0 0.6 0.5],'linewidth',2)
ylim([6.8 7.3])
xlim([1 n_iter])
xticks([1:5:n_iter])
legend('State-informed model-free','Observation-only model-free','Policy gradient','LSPI','location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('$L^{\pi}$','Interpreter','latex')



