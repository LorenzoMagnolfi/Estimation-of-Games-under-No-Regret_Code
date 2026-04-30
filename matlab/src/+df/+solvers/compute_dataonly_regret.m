function [R1, R2, mean_profit_1, mean_profit_2] = compute_dataonly_regret(cfg, distY, c1_cand, c2_cand)
% DF.SOLVERS.COMPUTE_DATAONLY_REGRET  Per-player max regret of an empirical
% action distribution under candidate costs (NST-style).
%
%   [R1, R2] = df.solvers.compute_dataonly_regret(cfg, distY, c1_cand, c2_cand)
%
%   For the realization-conditional restricted (data-only) spec, see the
%   addendum in JPE Revision/Realization_Identification_Design.md and the
%   NST 2015 connection note in JPE Revision/litreview/NST_2015.md.
%
%   For player i and candidate cost c_i = c{1,2}_cand:
%      R_i = max_{a_i' in A_i} sum_a distY(a) * [pi_i(a_i', a_{-i}; c_i) - pi_i(a; c_i)]
%
%   The candidate (c1_cand, c2_cand) is in the data-only identified set iff
%   R1 <= eps_1 and R2 <= eps_2 (see runner for the eps choice).
%
%   Inputs
%     cfg      game config from df.setup.game_simulation (logit demand assumed)
%     distY    (NActPr x 1) empirical action distribution m_N(a_1, a_2)
%     c1_cand  candidate marginal cost for player 1
%     c2_cand  candidate marginal cost for player 2
%
%   Outputs
%     R1, R2                max regret per player (scalar)
%     mean_profit_1, _2     mean profit under distY at the candidate cost (for diagnostics)

A         = cfg.A;
AA        = cfg.AA;
NPlayers  = cfg.NPlayers;
alpha_val = cfg.alpha;

NActPr = size(A, 1);
nAct   = size(AA, 1);

%% Player 1 regret
profits_1 = zeros(NActPr, 1);
for k = 1:NActPr
    a1 = A(k, 1); a2 = A(k, 2);
    cp_1 = exp(alpha_val * a1) / (1 + exp(alpha_val * a1) + exp(alpha_val * a2));
    profits_1(k) = cp_1 * (a1 - c1_cand);
end
mean_profit_1 = sum(distY .* profits_1);

deviation_profits_1 = zeros(nAct, 1);
for j = 1:nAct
    a1_alt = AA(j);
    for k = 1:NActPr
        a2 = A(k, 2);
        cp_alt = exp(alpha_val * a1_alt) / (1 + exp(alpha_val * a1_alt) + exp(alpha_val * a2));
        deviation_profits_1(j) = deviation_profits_1(j) + distY(k) * cp_alt * (a1_alt - c1_cand);
    end
end
R1 = max(deviation_profits_1) - mean_profit_1;

%% Player 2 regret
profits_2 = zeros(NActPr, 1);
for k = 1:NActPr
    a1 = A(k, 1); a2 = A(k, 2);
    cp_2 = exp(alpha_val * a2) / (1 + exp(alpha_val * a1) + exp(alpha_val * a2));
    profits_2(k) = cp_2 * (a2 - c2_cand);
end
mean_profit_2 = sum(distY .* profits_2);

deviation_profits_2 = zeros(nAct, 1);
for j = 1:nAct
    a2_alt = AA(j);
    for k = 1:NActPr
        a1 = A(k, 1);
        cp_alt = exp(alpha_val * a2_alt) / (1 + exp(alpha_val * a1) + exp(alpha_val * a2_alt));
        deviation_profits_2(j) = deviation_profits_2(j) + distY(k) * cp_alt * (a2_alt - c2_cand);
    end
end
R2 = max(deviation_profits_2) - mean_profit_2;

end
