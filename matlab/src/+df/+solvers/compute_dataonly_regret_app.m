function [R, kappa] = compute_dataonly_regret_app(cfg, c_cand)
% DF.SOLVERS.COMPUTE_DATAONLY_REGRET_APP  Per-player max regret of an empirical
% action distribution under candidate cost, using the application's data-driven
% sale-probability payoff structure.
%
%   [R, kappa] = df.solvers.compute_dataonly_regret_app(cfg, c_cand)
%
%   Application-side companion to df.solvers.compute_dataonly_regret (which
%   uses the simulation's logit-demand structure).  The application payoff is
%       pi(a, a_{-i}; c) = prob(a, a_{-i}) * (a - c)
%   where prob is the empirical sale probability indexed by the joint action
%   profile.
%
%   For the focal player (the one cfg was built for via
%   df.setup.game_application), the data-only regret-rationalizability test:
%      R(c_cand) = max_{a' in A} sum_{aa} m_N(aa) * [
%                       prob_dev(aa, a') * (a' - c_cand)
%                     - prob(aa) * (cfg.A(aa, 1) - c_cand) ]
%   where prob_dev(aa, a') is the sale probability if the focal player had
%   played a' instead of cfg.A(aa, 1) while the opponent played cfg.A(aa, 2).
%
%   Inputs
%     cfg     - application config from df.setup.game_application
%     c_cand  - candidate marginal cost for the focal player
%
%   Outputs
%     R       - max regret (scalar)
%     kappa   - payoff range max-min over (a, a_{-i}) at this cost; for the
%               candidate-specific eps_R1.

A = cfg.A;            % (NActPr x 2) action profiles
AA = cfg.AA;          % (NAct x 1) own-action grid
prob = cfg.prob;      % (NActPr x 1) sale probability per profile
distrib = cfg.distrib(:);   % (NActPr x 1) empirical action distribution m_N

NActPr = size(A, 1);
NAct = size(AA, 1);

% Build action-profile lookup: idx_lookup(a_self_idx, a_opp_idx) -> profile index
idx_lookup = zeros(NAct, NAct);
for k = 1:NActPr
    a1_idx = find(AA == A(k, 1), 1);
    a2_idx = find(AA == A(k, 2), 1);
    idx_lookup(a1_idx, a2_idx) = k;
end

% Baseline (own-played action) payoff per profile
baseline_payoff = prob .* (A(:, 1) - c_cand);     % (NActPr x 1)
baseline_mean = sum(distrib .* baseline_payoff);

% Deviation: for each alternative own action a' in AA, integrate over m_N
deviation_means = zeros(NAct, 1);
for j = 1:NAct
    a_self = AA(j);
    dev_payoff_per_prof = zeros(NActPr, 1);
    for k = 1:NActPr
        a_opp_idx = find(AA == A(k, 2), 1);
        prof_idx_dev = idx_lookup(j, a_opp_idx);
        dev_payoff_per_prof(k) = prob(prof_idx_dev) * (a_self - c_cand);
    end
    deviation_means(j) = sum(distrib .* dev_payoff_per_prof);
end

R = max(deviation_means) - baseline_mean;

% Kappa at this candidate: payoff range over (a_self, a_opp) at cost c_cand
all_payoffs = zeros(NActPr, 1);
for k = 1:NActPr
    all_payoffs(k) = prob(k) * (A(k, 1) - c_cand);
end
kappa = max(all_payoffs) - min(all_payoffs);

end
