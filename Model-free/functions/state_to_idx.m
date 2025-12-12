function [idx] = state_to_idx(s,a,t,nu,T)
idx = (s-1)*nu*T + (a-1)*T + t;
end