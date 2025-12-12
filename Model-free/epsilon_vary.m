clear all; close all; clc; warning off;
addpath('functions')
rng(0,'twister');

%% General settings
nx = 5;  
ny = 5;  
nu = 5;
T = 10;

n_iter = 31;
n_simulations = 5000;

epsilon_vec = [0.01,0.25,0.5,0.75,1];     
n_epsilon = length(epsilon_vec);
n_MC = 20;

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

%% State-informed model-free policy iteration
SIMF.n_iter = n_iter;
SIMF.ns_DATA = n_simulations;                % number of episodes per dataset (state-informed)
SIMF.n_sweeps = 1;         % how many forward+backward sweeps per dataset

SIMF.J_test = zeros(n_epsilon,n_MC,SIMF.n_iter);

for eps = 1:n_epsilon
epsilon_behave = epsilon_vec(eps);

for MC = 1:n_MC
SIMF.policy = cell(SIMF.n_iter+1,1);
SIMF.policy{1} = pi_0;
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
    SIMF.J_test(eps,MC,iter) = evaluate_policy_td(P, O, g, Vt, SIMF.policy{iter+1}, mu_0);
    fprintf('Iter %d done. J_test = %.6g\n', iter, SIMF.J_test(eps,MC,iter));

end % iter loop
end
end

%% Figures
figure(1)
plot(mean(squeeze(SIMF.J_test(1,:,:)),1),'k','linewidth',2)
hold on; grid on;
plot(mean(squeeze(SIMF.J_test(2,:,:)),1),'color',[0.9 0.6 0],'linewidth',2)
plot(mean(squeeze(SIMF.J_test(3,:,:)),1),'color',[0.35 0.7 0.9],'linewidth',2)
plot(mean(squeeze(SIMF.J_test(4,:,:)),1),'color',[0 0.6 0.5],'linewidth',2)
plot(mean(squeeze(SIMF.J_test(5,:,:)),1),'color',[0.8 0.4 0],'linewidth',2)
ylim([6.95 7.25])
xlim([1 n_iter])
xticks([1:5:n_iter])
legend('$\epsilon = 0.01$','$\epsilon = 0.25$','$\epsilon = 0.50$','$\epsilon = 0.75$','$\epsilon = 1$','Interpreter','latex','location','southeast')
xlabel('Iteration','interpreter','latex')
ylabel('$L^{\pi}$','Interpreter','latex')


