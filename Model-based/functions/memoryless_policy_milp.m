function [policy, J1, solinfo] = memoryless_policy_milp_v3(P, O, g, gT, pi0, T, opts)
% MEMORYLESS_POLICY_MILP_FAST  Fast preallocated assembler for the MILP
%   Same interface as before. Supports opts.solver = 'intlinprog' (default)
%   or 'gurobi' (requires Gurobi MATLAB interface installed).
%
%   Usage:
%     opts.solver = 'gurobi';
%     opts.timeLimit = 60;  % seconds (optional)
%     [policy, J1, solinfo] = memoryless_policy_milp_fast(P,O,g,gT,pi0,T,opts);
%
% Outputs:
%   policy : (T+1) x ny deterministic actions (1..nu)
%   J1     : expected first-stage cost
%   solinfo: solver information struct

if nargin < 7, opts = struct(); end
if ~isfield(opts,'solver'), opts.solver = 'intlinprog'; end
if ~isfield(opts,'timeLimit'), opts.timeLimit = []; end

[nx, ~, nu] = size(P);
[ny, nx2] = size(O);
assert(nx2 == nx, 'O must be ny x nx');
assert(size(g,1) == T+1, 'g must have T+1 stages (t=0..T)');
pi0 = pi0(:);

% variable counts (same ordering as previous implementation)
n_mu_s   = (T+1)*nx;
n_mu_soa = (T+1)*nx*ny*nu;
n_mu_sa  = (T+1)*nx*nu;
n_delta  = (T+1)*nu*ny;
nv = n_mu_s + n_mu_soa + n_mu_sa + n_delta;

% index helpers
idx_mu_s = @(t,s)      ( (t)*nx + s );                      % t in 0..T, s in 1..nx
idx_mu_soa = @(t,s,y,a) ( n_mu_s + (t)*(nx*ny*nu) + ( (s-1)*ny*nu + (y-1)*nu + a ) );
idx_mu_sa = @(t,s,a)   ( n_mu_s + n_mu_soa + (t)*(nx*nu) + ( (s-1)*nu + a ) );
idx_delta = @(t,a,y)   ( n_mu_s + n_mu_soa + n_mu_sa + (t)*(nu*ny) + ( (a-1)*ny + y ) );

% -------------------------
% Objective
% -------------------------
f = zeros(nv,1);
for t = 0:T
    gt = squeeze(g(t+1,:,:));   % nx x nu (MATLAB indexing t+1)
    for s = 1:nx
        base = idx_mu_sa(t,s,1);
        f(base:base+nu-1) = -gt(s,:).'; % minimize negative objective
    end
end

% Add terminal cost gT(s) applied to mu_{T,s}
for s = 1:nx
    f(idx_mu_s(T, s)) = f(idx_mu_s(T, s)) - gT(s);
end

% -------------------------
% Bounds and intcon
% -------------------------
lb = zeros(nv,1);
ub = inf(nv,1);

% mu variables probabilities <=1
for t = 0:T
    for s = 1:nx
        ub(idx_mu_s(t,s)) = 1;
    end
    for s = 1:nx
        for y = 1:ny
            for a = 1:nu
                ub(idx_mu_soa(t,s,y,a)) = 1;
            end
        end
    end
    for s = 1:nx
        for a = 1:nu
            ub(idx_mu_sa(t,s,a)) = 1;
        end
    end
end

% delta bounds 0..1
for t = 0:T
    for a = 1:nu
        for y = 1:ny
            ub(idx_delta(t,a,y)) = 1;
        end
    end
end

% integer indices: all delta variables (binary)
intcon = zeros(n_delta,1);
k = 0;
for t = 0:T
    for a = 1:nu
        for y = 1:ny
            k = k + 1;
            intcon(k) = idx_delta(t,a,y);
        end
    end
end

% -------------------------
% Count constraints to preallocate triplet arrays
% -------------------------
n_eq = 0;
n_eq = n_eq + nx;                         % mu0 constraints
n_eq = n_eq + (T+1)*nx;                   % mu_t_s = sum_a mu_t_sa
n_eq = n_eq + (T+1)*nx*nu;                % mu_t_sa = sum_o mu_t_soa
n_eq = n_eq + (T)*nx;                     % dynamics mu_{t+1}
n_eq = n_eq + nu*(ny-1);                  % delta0 equalities

n_ineq = 0;
n_ineq = n_ineq + 3*(T+1)*nx*ny*nu;      % McCormick (9a,9b,9c)

% rough nnz estimates
nnz_per_eq =  max(10, nu + ny*nu + 5);
nnz_eq_est = n_eq * nnz_per_eq;
nnz_per_ineq = 4;
nnz_ineq_est = n_ineq * nnz_per_ineq;

% Preallocate triplet arrays
i_eq = zeros(max(1,nnz_eq_est),1);
j_eq = zeros(max(1,nnz_eq_est),1);
v_eq = zeros(max(1,nnz_eq_est),1);
eq_nnz = 0;

i_ineq = zeros(max(1,nnz_ineq_est),1);
j_ineq = zeros(max(1,nnz_ineq_est),1);
v_ineq = zeros(max(1,nnz_ineq_est),1);
ineq_nnz = 0;

beq = zeros(n_eq,1);
bineq = zeros(n_ineq,1);

% helpers for pushing triplets (fast growth with doubling)
    function push_eq(r,col,val)
        eq_nnz = eq_nnz + 1; %#ok<NASGU>
        if eq_nnz > length(i_eq)
            extend = 2*length(i_eq);
            i_eq(end+1:extend) = 0;
            j_eq(end+1:extend) = 0;
            v_eq(end+1:extend) = 0;
        end
        i_eq(eq_nnz) = r;
        j_eq(eq_nnz) = col;
        v_eq(eq_nnz) = val;
    end

    function push_ineq(r,col,val)
        ineq_nnz = ineq_nnz + 1; %#ok<NASGU>
        if ineq_nnz > length(i_ineq)
            extend = 2*length(i_ineq);
            i_ineq(end+1:extend) = 0;
            j_ineq(end+1:extend) = 0;
            v_ineq(end+1:extend) = 0;
        end
        i_ineq(ineq_nnz) = r;
        j_ineq(ineq_nnz) = col;
        v_ineq(ineq_nnz) = val;
    end

% -------------------------
% Build equality constraints
% -------------------------
r = 0; % equality row counter

% (7c) mu0_s = pi0(s)
for s = 1:nx
    r = r + 1;
    push_eq(r, idx_mu_s(0,s), 1);
    beq(r) = pi0(s);
end

% (7d) mu_t_s = sum_a mu_t_sa  for all t,s
for t = 0:T
    for s = 1:nx
        r = r + 1;
        push_eq(r, idx_mu_s(t,s), 1);
        for a = 1:nu
            push_eq(r, idx_mu_sa(t,s,a), -1);
        end
        beq(r) = 0;
    end
end

% (7e) mu_t_sa = sum_o mu_t_soa for all t,s,a
for t = 0:T
    for s = 1:nx
        for a = 1:nu
            r = r + 1;
            push_eq(r, idx_mu_sa(t,s,a), 1);
            for y = 1:ny
                push_eq(r, idx_mu_soa(t,s,y,a), -1);
            end
            beq(r) = 0;
        end
    end
end

% (7f) dynamics mu_{t+1}_s = sum_{sp,ap} p(s|sp,ap) * mu_t_sp_ap   for t=0..T-1
for t = 0:(T-1)
    for s = 1:nx
        r = r + 1;
        push_eq(r, idx_mu_s(t+1,s), 1);
        for sp = 1:nx
            for ap = 1:nu
                coeff = P(sp,s,ap);
                if coeff ~= 0
                    push_eq(r, idx_mu_sa(t,sp,ap), -coeff);
                end
            end
        end
        beq(r) = 0;
    end
end

% (7b) delta0 equal across observations
for a = 1:nu
    for y = 2:ny
        r = r + 1;
        push_eq(r, idx_delta(0,a,y), 1);
        push_eq(r, idx_delta(0,a,1), -1);
        beq(r) = 0;
    end
end

% trim beq if necessary
if r ~= n_eq
    beq = beq(1:r);
    n_eq = r;
end

% -------------------------
% Build McCormick inequalities (9a,9b,9c)
% -------------------------
ir = 0;
for t = 0:T
    for s = 1:nx
        for y = 1:ny
            p_o_s = O(y,s);
            for a = 1:nu
                % (9a) mu_t_soa - p(o|s)*mu_t_s <= 0
                ir = ir + 1;
                push_ineq(ir, idx_mu_soa(t,s,y,a), 1);
                if p_o_s ~= 0
                    push_ineq(ir, idx_mu_s(t,s), -p_o_s);
                end
                bineq(ir) = 0;
                % (9b) mu_t_soa - delta_t_a_o <= 0
                ir = ir + 1;
                push_ineq(ir, idx_mu_soa(t,s,y,a), 1);
                push_ineq(ir, idx_delta(t,a,y), -1);
                bineq(ir) = 0;
                % (9c) -mu_t_soa + p(o|s)*mu_t_s + delta_t_a_o <= 1
                ir = ir + 1;
                push_ineq(ir, idx_mu_soa(t,s,y,a), -1);
                if p_o_s ~= 0
                    push_ineq(ir, idx_mu_s(t,s), p_o_s);
                end
                push_ineq(ir, idx_delta(t,a,y), 1);
                bineq(ir) = 1;
            end
        end
    end
end

if ir ~= n_ineq
    bineq = bineq(1:ir);
    n_ineq = ir;
end

% Trim triplet arrays to actual sizes
i_eq = i_eq(1:eq_nnz); j_eq = j_eq(1:eq_nnz); v_eq = v_eq(1:eq_nnz);
i_ineq = i_ineq(1:ineq_nnz); j_ineq = j_ineq(1:ineq_nnz); v_ineq = v_ineq(1:ineq_nnz);

% Build sparse matrices
Aeq = sparse(i_eq, j_eq, v_eq, n_eq, nv);
Aineq = sparse(i_ineq, j_ineq, v_ineq, n_ineq, nv);

% -------------------------
% Solve MILP using selected solver
% -------------------------
if strcmpi(opts.solver,'intlinprog')
    ipopts = optimoptions('intlinprog','Display','off');
    if ~isempty(opts.timeLimit)
        ipopts.TimeLimit = opts.timeLimit;
    end
    tic;
    try
        [x,fval,exitflag,output] = intlinprog(f,intcon,Aineq,bineq,Aeq,beq,lb,ub,ipopts);
    catch ME
        error('intlinprog failed: %s', ME.message);
    end
    soltime = toc;
    solinfo.exitflag = exitflag;
    solinfo.output = output;
    solinfo.solver = 'intlinprog';
    solinfo.obj = -fval;
    solinfo.time = soltime;

elseif strcmpi(opts.solver,'gurobi')
    % Build Gurobi model struct
    % Gurobi minimizes by default, so keep f as is (we already negated objective)
    % model.A is the stacked constraints Aeq; Aineq
    A = [Aeq; Aineq];            % (n_eq + n_ineq) x nv sparse
    rhs = [beq; bineq];         % corresponding rhs
    sense = [repmat('=', n_eq,1); repmat('<', n_ineq,1)]; % char vector

    model.A = A;
    model.obj = f;
    model.rhs = rhs;
    model.sense = sense;
    model.lb = lb;
    model.ub = ub;

    % vtype: default continuous then set binary for intcon
    vtype = repmat('C', nv, 1);
    vtype(intcon) = 'B';
    model.vtype = vtype;

    % Gurobi params
    params = struct();
    params.OutputFlag = 0;
    if ~isempty(opts.timeLimit)
        params.TimeLimit = opts.timeLimit;
    end
    params.Presolve     = 2;      % aggressive presolve
params.Cuts         = 2;      % aggressive cut generation
params.Heuristics   = 0.2;    % early heuristic solutions
params.MIPFocus     = 1;      % focus on finding feasible quickly
params.Threads      = max(1, feature('numcores')-1);
params.Method       = 1;      % primal simplex for root relaxation
params.BarConvTol   = 1e-8;
params.NumericFocus = 1;
params.ScaleFlag    = 2;
params.OutputFlag   = 0;

    % solve
    tic;
    try
        result = gurobi(model, params);
    catch ME
        error('Gurobi call failed: %s', ME.message);
    end
    soltime = toc;

    % interpret result
    solinfo.solver = 'gurobi';
    solinfo.obj = [];
    solinfo.time = soltime;
    solinfo.status = result.status;
    if isfield(result,'objval'), solinfo.obj = -result.objval; end
    if strcmp(result.status,'OPTIMAL') || strcmp(result.status,'TIME_LIMIT')
        if isfield(result,'x') && ~isempty(result.x)
            x = result.x;
        else
            % no primal solution returned
            error('Gurobi did not return a solution vector (result.x empty).');
        end
    else
        error('Gurobi returned status: %s', result.status);
    end

else
    error('Unknown solver %s', opts.solver);
end

x = x(:);

% Extract deterministic policy from delta variables
policy = zeros(T+1, ny);
for t = 0:T
    for y = 1:ny
        best_val = -inf;
        best_a = 1;
        for a = 1:nu
            v = x(idx_delta(t,a,y));
            if v > best_val
                best_val = v;
                best_a = a;
            end
        end
        policy(t+1,y) = best_a;
    end
end

% Compute expected first-stage cost J1 using mu_t_sa variables (t=0)
% J1 = 0;
% for s = 1:nx
%     for a = 1:nu
%         pos = idx_mu_sa(0,s,a);
%         J1 = J1 + g(1,s,a) * x(pos);
%     end
% end
% % terminal cost at time T
% for s = 1:nx
%     J1 = J1 + gT(s) * x(idx_mu_s(T,s));
% end

% initialize state distribution at time 0
mu = pi0(:);            % nx x 1

J1 = 0;

for t = 0:T
    % Build probability of choosing each action given state s at time t:
    % Aprob(s,a) = sum_{y : policy(t,y)=a} O(y|s)
    Aprob = zeros(nx, nu);
    for s = 1:nx
        for y = 1:ny
            a = policy(t+1, y);      % action chosen when observation = y at time t
            Aprob(s, a) = Aprob(s, a) + O(y, s);
        end
    end

    % Compute joint probability mu_{t,s,a} = mu_t(s) * Pr(a | s, policy)
    mu_sa = zeros(nx, nu);
    for s = 1:nx
        for a = 1:nu
            mu_sa(s,a) = mu(s) * Aprob(s,a);
            % accumulate stage cost g(t+1, s, a)  (MATLAB indexing)
            J1 = J1 + g(t+1, s, a) * mu_sa(s,a);
        end
    end

    % propagate state distribution to next time (unless last iteration)
    if t < T
        mu_next = zeros(nx,1);
        % P(sp, s, a) = Prob(s_{t+1} = s | s_t = sp, a_t = a)
        for sp = 1:nx
            for a = 1:nu
                if mu_sa(sp,a) == 0
                    continue;
                end
                % add contribution from all next-states s
                % note index order P(prev, next, action)
                for s = 1:nx
                    coeff = P(sp, s, a);
                    if coeff ~= 0
                        mu_next(s) = mu_next(s) + coeff * mu_sa(sp,a);
                    end
                end
            end
        end
        mu = mu_next;
    else
        % at t == T we also add terminal cost gT(s) applied to mu_T(s)
        for s = 1:nx
            J1 = J1 + gT(s) * mu(s);
        end
    end
end

end