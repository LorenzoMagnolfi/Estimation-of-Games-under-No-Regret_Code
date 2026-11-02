function eps_t = compute_epsilon_v05(cfg, N, phi, r_cons, switch_eps, alpha_R)
% DF.SOLVERS.COMPUTE_EPSILON_V05  Per-type obedience tolerances from the V05 calendar inputs.
%
%   eps_t = df.solvers.compute_epsilon_v05(cfg, N, phi, r_cons, switch_eps, alpha_R)
%
%   Returns the 1 x s vector of per-period obedience tolerances (units of average
%   regret) at ONE candidate parameter, for the finite-sample calibrations of
%   Niccolo's V05 manuscript (2026-11-02, Section 4 and Appendix "Learning-Design
%   Foundations"). Unlike df.solvers.compute_epsilon, whose radius is multiplied by
%   sqrt(phi) at the LP layer, these tolerances depend on the candidate's type shares
%   through the latent context count and are therefore computed per candidate.
%
%   Inputs
%     cfg        game config (.Pi, .NAct, .NPlayers, .s)
%     N          horizon (number of periods)
%     phi        1 x s candidate marginal type probabilities (players symmetric)
%     r_cons     L1 consistency radius at this N (0 when the composition is exact)
%     switch_eps 14 tailF   : full-feedback family F, direct tail at delta = alpha_R / M_R,
%                             count ceiling m = N min{1, phi + r_cons/2} (valid on the
%                             consistency event; weakly looser than the witness count)
%                15 tailB   : stabilized bandit family B, direct tail, same count ceiling
%                16 momentF : family F first moment at the expected count N phi, Markov
%                             rectangle with equal weights (r_k <= M_R b_k / alpha_R)
%                17 momentB : family B first moment, same rectangle
%     alpha_R    obedience error budget (the composition budget alpha_C is separate)
%
%   Constants (V05 eqs. s4Finputs, s4Bmoment, s4Btail): Hedge of df.sim.learn_hedge has
%   step multiplier c = 1, so c_F = 1.25 sqrt(ln d); EXP3-IX of df.sim.learn_exp3ix has
%   bands a = 1, b = 1/2, so A = 3, b_- = 1/2, C_pr = ln d; K_t is the payoff span of
%   the type (the full-feedback span bound and the bandit interval width L_t);
%   M_R = NPlayers * s contextual components; J_N = 1 + ceil(log2 N).

d = cfg.NAct;
s = cfg.s;
M_R = cfg.NPlayers * s;
delta = alpha_R / M_R;
K = max(cfg.Pi(:, :, 1), [], 1) - min(cfg.Pi(:, :, 1), [], 1);   % 1 x s payoff span
phi = reshape(phi, 1, []);
if numel(phi) ~= s
    error('compute_epsilon_v05:BadPhi', 'phi must have %d entries (got %d).', s, numel(phi));
end
JN = 1 + ceil(log2(N));
cF = 1.25 * sqrt(log(d));
Afam = 3; bminus = 0.5; C = log(d);

switch switch_eps
    case 14
        m = N * min(1, phi + r_cons / 2);
        H = K .* min(m, cF * sqrt(m) + sqrt((m + 1) * log(JN / delta)));
        eps_t = H / N;
    case 15
        m = N * min(1, phi + r_cons / 2);
        H = K .* min(m, Afam * sqrt(d * m * C) + (sqrt(2) / (2 * bminus)) * sqrt(d * m / C) * log(2 * d * JN / delta) + log(2 / delta));
        eps_t = H / N;
    case 16
        mu = N * phi;
        b = K .* min(mu, (cF + 0.25) * sqrt(mu)) / N;
        eps_t = (M_R / alpha_R) * b;
    case 17
        mu = N * phi;
        b = K .* min(mu, Afam * sqrt(d * C * mu) + (log(d) + 1) / (2 * bminus) * sqrt(d * N / C)) / N;
        eps_t = (M_R / alpha_R) * b;
    case {18, 19}
        % Noisy-feedback bridge (V05, Appendix "Noisy Reward Feedback", calendar form): the learner's
        % guarantee holds for the feedback regret on potential profits in a known interval of width
        % L_t (cash units); structural regret adds at most 2 L sqrt((N_k + 1) ln(d J_N / delta_D)) with
        % error delta_D. Tail at delta/2, bridge at delta/2; cap K m (expected-profit units).
        %   18: full potential-feedback vector, Hedge on Y (family F with span L)      [learn_hedge_noisy]
        %   19: selected feedback only, EXP3-IX on Y (family B with width L)           [learn_exp3ix_noisy]
        c_t = cfg.tps(:, 1)';                                         % cost of each type (players symmetric)
        L = max(max(cfg.AA) - c_t, 0) + max(c_t - min(cfg.AA), 0);   % realized-profit interval width, cash units
        m = N * min(1, phi + r_cons / 2);
        d2 = delta / 2;
        bridge = 2 * L .* sqrt((m + 1) * log(d * JN / d2));
        if switch_eps == 18
            core = L .* (cF * sqrt(m) + sqrt((m + 1) * log(JN / d2)));
        else
            core = L .* (Afam * sqrt(d * m * C) + (sqrt(2) / (2 * bminus)) * sqrt(d * m / C) * log(2 * d * JN / d2) + log(2 / d2));
        end
        eps_t = min(K .* m, core + bridge) / N;
    otherwise
        error('compute_epsilon_v05:BadSwitch', 'switch_eps must be 14 to 19 (got %d).', switch_eps);
end
eps_t = max(eps_t, 0);
end
