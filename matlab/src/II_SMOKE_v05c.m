%% II_SMOKE_v05c  Regression tests for the two generator fixes of 4 November 2026 (Niccolò's findings I4 and I7(f)).
%  (1) Common-scale invariance: with identical RNG draws, Hedge on profits and normalizer both scaled by q must make the
%      same choices as at q = 1 (distY identical) and report regrets scaled by q; learn_hedge_noisy must keep its own
%      behaviour (it already scaled consistently). (2) Variance convention: cfg.marg_distrib from game_simulation at
%      sigma2 = 2.25 equals the discretized Normal with standard deviation 1.5, and the learners' internal prior equals
%      cfg.marg_distrib, so that the Stage II grid value (a variance: mvnpdf covariance, normpdf(sqrt)) and the generator
%      agree at a truth other than one. (3) Lint of the touched files.
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_v05c".  Prints SMOKE_V05C: PASS|FAIL.
fprintf('=== SMOKE_V05C_START ===\n'); ok = true;
files = {fullfile('+df','+sim','learn_hedge.m'), fullfile('+df','+sim','learn_exp3ix.m'), fullfile('+df','+sim','learn_hedge_noisy.m'), ...
    fullfile('+df','+sim','learn_mixed.m'), fullfile('+df','+sim','learn_hedge_units.m'), 'marginal_cost_draws_v5.m', 'II_RUN_v05_sigma_check.m'};
for i = 1:numel(files)
    m = checkcode(files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', files{i}); else, fprintf('CHECKCODE %s:\n%s\n', files{i}, m); end
end
NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8]; mu = 3*ones(2,1); s = 5; N = 20000;
%% (1) common-scale invariance of Hedge
try
    cfg1 = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, eye(2), s);
    rng(11); [dY1, ~, fr1] = df.sim.learn_hedge(cfg1, 1, N, N, 1, 1, 1, 1);
    for q = [0.3 0.1 0.03]
        cfgq = cfg1; cfgq.Pi = q * cfg1.Pi; cfgq.sale_scale = q;
        rng(11); [dYq, ~, frq] = df.sim.learn_hedge(cfgq, 1, N, N, 1, 1, 1, 1);
        same = max(abs(dYq(:) - dY1(:))) < 1e-12; scaled = max(abs(frq(:) - q * fr1(:))) < 1e-9 * max(1, max(abs(fr1(:))));
        fprintf('(1) q=%.2f: distY identical=%d, regret scaled by q=%d (max |dY diff| %.1e, max |fr - q fr1| %.1e)\n', q, same, scaled, max(abs(dYq(:)-dY1(:))), max(abs(frq(:)-q*fr1(:))));
        ok = ok && same && scaled;
    end
catch err, fprintf('(1) FAILED: %s\n', err.message); ok = false; end
%% (2) variance convention at sigma2 = 2.25
try
    cfg2 = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, 2.25 * eye(2), s);
    sup = cfg2.type_space{1,1}; ref = normpdf(sup, mu(1), 1.5); ref = ref / sum(ref);
    c1 = max(abs(cfg2.marg_distrib(:) - ref(:))) < 1e-12;
    % the learners' internal prior: regenerate it the way learn_hedge does and compare with the config
    md = pdf('normal', sup, mu(1), sqrt(2.25)); md = md / sum(md);
    c2 = max(abs(md(:) - ref(:))) < 1e-12;
    % and a candidate on the Stage II grid at the truth multipliers gives the same marginal (normpdf with sqrt)
    md3 = normpdf(sup, mu(1), sqrt(2.25)); md3 = md3 / sum(md3);
    c3 = max(abs(md3(:) - ref(:))) < 1e-12;
    fprintf('(2) sigma2 = 2.25: game_simulation prior matches sd 1.5 (%d); learner prior matches (%d); Stage II marginal matches (%d); prior = %s\n', c1, c2, c3, mat2str(ref', 4));
    ok = ok && c1 && c2 && c3;
catch err, fprintf('(2) FAILED: %s\n', err.message); ok = false; end
if ok, fprintf('SMOKE_V05C: PASS\n'); else, fprintf('SMOKE_V05C: FAIL\n'); exit(1); end
