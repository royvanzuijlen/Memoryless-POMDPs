function [J, grad] = modelGrad(z, fun)
    J = fun(z);             % scalar output
    grad = dlgradient(J, z); % gradient w.r.t z
end