function Ppi = induced_P(P, O, mu)
    [nx, ~, nu] = size(P);
    ny = length(mu);
    % compute Prob(u | i)
    ProbUgivenI = zeros(nx, nu);
    for y = 1:ny
        u = mu(y);
        ProbUgivenI(:,u) = ProbUgivenI(:,u) + O(y,:)' ; % add O(y,i) to column for action u
    end
    % Now form Ppi
    Ppi = zeros(nx, nx);
    for u = 1:nu
        % multiply each row i of P(:,:,u) by ProbUgivenI(i,u) and sum over u
        Pu = squeeze(P(:,:,u));          % nx x nx
        % elementwise multiply each row of Pu by ProbUgivenI(:,u)
        Ppi = Ppi + bsxfun(@times, ProbUgivenI(:,u), Pu);
    end
end