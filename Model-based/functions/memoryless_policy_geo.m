function [policy, J, solinfo] = memoryless_policy_geo(P,O,g,gT,pi0,T,opts)
% MEMORYLESS_STOCHASTIC_POLICY_FH
% Compute optimal finite-horizon MEMORYLESS STOCHASTIC policy (maximize reward)
% Solver options:
%   opts.solver = 'fmincon' (default)
%   opts.solver = 'gurobi'  (requires nonlinear General Constraints enabled)

if nargin < 7, opts = struct(); end
if ~isfield(opts,'solver'), opts.solver = 'fmincon'; end

[nx,~,nu] = size(P);
[ny,nx2] = size(O);
assert(nx2 == nx);

pi0 = pi0(:);

nvar = (T+1)*ny*nu;
idx = @(t,y,a) (t*ny*nu + (y-1)*nu + a);

x0 = ones(nvar,1)/nu;

%% equality constraints: sum_a π_{t,y,a} = 1
Aeq = zeros((T+1)*ny,nvar);
beq = ones((T+1)*ny,1);
row = 0;
for t = 0:T
    for y = 1:ny
        row = row + 1;
        for a = 1:nu
            Aeq(row,idx(t,y,a)) = 1;
        end
    end
end

lb = zeros(nvar,1);
ub = ones(nvar,1);

%% expected reward function
    function R = expected_reward(x)
        pi_t2 = zeros(T+1,ny,nu);
        k = 0;
        for t = 0:T
            for y = 1:ny
                pi_t2(t+1,y,:) = x(k+1:k+nu);
                k = k + nu;
            end
        end

        mu = pi0;
        R = 0;

        for t = 0:T
            Aprob = zeros(nx,nu);
            for s = 1:nx
                for y = 1:ny
                    Oy = O(y,s);
                    if Oy==0, continue; end
                    for a = 1:nu
                        Aprob(s,a) = Aprob(s,a) + Oy*pi_t2(t+1,y,a);
                    end
                end
            end

            for s = 1:nx
                for a = 1:nu
                    R = R + mu(s)*Aprob(s,a)*g(t+1,s,a);
                end
            end

            if t < T
                mu_next = zeros(nx,1);
                for sp = 1:nx
                    for a = 1:nu
                        pa = Aprob(sp,a);
                        if pa==0, continue; end
                        for s = 1:nx
                            mu_next(s) = mu_next(s) + mu(sp)*pa*P(sp,s,a);
                        end
                    end
                end
                mu = mu_next;
            else
                R = R + sum(mu .* gT(:));
            end
        end
    end

%% Case 1: fmincon (default)
if strcmpi(opts.solver,'fmincon')

    obj = @(x) -expected_reward(x);

    options = optimoptions('fmincon','Display','iter','Algorithm','sqp');
    if isfield(opts,'timeLimit') && ~isempty(opts.timeLimit)
        options.MaxTime = opts.timeLimit;
    end

    tic;
    [x_opt,fval,exitflag,output] = fmincon(obj,x0,[],[],Aeq,beq,lb,ub,[],options);
    soltime = toc;

    solinfo.solver = 'fmincon';
    solinfo.fval = fval;
    solinfo.exitflag = exitflag;
    solinfo.output = output;
    solinfo.time = soltime;

%% Case 2: Gurobi nonlinear general constraints
elseif strcmpi(opts.solver,'gurobi')

    % 1 reward scalar + nvar policy variables
    model.vtype = repmat('C',nvar+1,1);

    Rvar = nvar+1;

    model.lb = [lb; -inf];
    model.ub = [ub; inf];

    % Linear constraints for sum_a pi = 1
    model.A = sparse([Aeq, zeros(size(Aeq,1),1)]);
    model.rhs = beq;
    model.sense = repmat('=',size(Aeq,1),1);

    % nonlinear general constraint: R == expected_reward(x)
    % we use a "general function" constraint
    model.gencon = struct();
    model.gencon(1).type = 'func';
    model.gencon(1).f = @(z) expected_reward(z(1:end-1)) - z(end);
    model.gencon(1).vars = 1:(nvar+1);

    % objective minimize -R
    model.obj = zeros(nvar+1,1);
    model.obj(Rvar) = -1;
    model.modelsense = 'min';

    params.OutputFlag = 1;
    if isfield(opts,'timeLimit') && ~isempty(opts.timeLimit)
        params.TimeLimit = opts.timeLimit;
    end

    result = gurobi(model,params);

    if ~strcmp(result.status,'OPTIMAL') && ~strcmp(result.status,'TIME_LIMIT')
        error('Gurobi failed: %s', result.status);
    end

    x_opt = result.x(1:nvar);

    solinfo.solver = 'gurobi';
    solinfo.status = result.status;
    solinfo.result = result;

else
    error('Unknown solver %s',opts.solver);
end

%% Extract policy and true reward
policy = zeros(T+1,ny,nu);
for t=0:T
    for y=1:ny
        for a=1:nu
            policy(t+1,y,a) = x_opt(idx(t,y,a));
        end
    end
end

J = expected_reward(x_opt);

end