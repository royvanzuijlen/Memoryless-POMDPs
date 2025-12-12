function v = one_hot_phi(t,o,a,T,ny,nu)
    % returns column vector of length T*ny*nu with a single 1 at (t,o,a)
    d = T * ny * nu;
    v = zeros(d,1);
    % linear index: ((t-1)*ny + (o-1))*nu + a
    idx = (t-1)*ny*nu + (o-1)*nu + a;
    v(idx) = 1;
end
