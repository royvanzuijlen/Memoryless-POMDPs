function [prob] = sm_pol_prob(s,t,nu,T,PG)
prob_unnorm = zeros(nu,1);
for i=1:nu
    feature_idx = state_to_idx(s,i,t,nu,T);
    feature_vec = PG.Phi(:,feature_idx);
    h = feature_vec'*PG.theta;
    prob_unnorm(i) = exp(h);
end
prob = prob_unnorm./sum(prob_unnorm);

end
