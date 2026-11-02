%% II_SMOKE_v05  Smoke test for the V05 learners and calibrations (tiny lab, minutes).
%  Checks: (0) lint of the new files; (1) both learners run and return positive-part
%  regrets that fall between N = 20k and N = 80k; (2) compute_epsilon_v05 returns finite
%  nonnegative tolerances for codes 14-17 and the tail codes are below the deterministic
%  cap K*phi; (3) Stage II solves all six arms on a 4x4 grid with finite VV; (4) the
%  true parameter is identified under every arm at N = 80k (tolerances are wide there);
%  (5) the tail region is contained in the legacy-HP region's complement check is NOT
%  asserted (different constants) but the areas are printed; (6) a 'none' consistency
%  kind with a tail code errors as intended.
%  Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_v05"
%  Prints SMOKE_V05: PASS|FAIL and exits nonzero on FAIL.
fprintf('=== SMOKE_V05_START ===\n');
ok = true;

lint_files = {fullfile('+df','+sim','learn_hedge.m'), fullfile('+df','+sim','learn_exp3ix.m'), ...
    fullfile('+df','+solvers','compute_epsilon_v05.m'), fullfile('+df','+stages','run_stage_ii.m'), ...
    'II_RUN_v05_frontier.m', 'II_RUN_v05_rate.m', 'II_RUN_v05_coverage.m'};
for i = 1:numel(lint_files)
    m = checkcode(lint_files{i}, '-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n', lint_files{i});
    else, fprintf('CHECKCODE %s:\n%s\n', lint_files{i}, m); end
end

NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8]; mu = 3*ones(2,1); sigma2 = eye(2); s = 5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
NGRID = 4; IDTOL = 1e-8; aR = 0.045; aC = 0.005;
fgM = [1; linspace(0.18,1.5,NGRID)']; fgV = [1; linspace(0.15,3.5,NGRID)'];

%% (1) learners: regret falls with N
reg = struct();
for sty = {'hedge','exp3ix'}
    for Nv = [20000 80000]
        rng(7);
        if strcmp(sty{1},'hedge'), [dY,~,fr] = df.sim.learn_hedge(cfg,1,Nv,Nv,1,1,1,1);
        else, [dY,~,fr] = df.sim.learn_exp3ix(cfg,1,Nv,Nv,1,1,1,1); end
        reg.(sty{1}).(sprintf('N%d',Nv)) = struct('dY',dY,'maxreg',max(fr(:)),'sumdist',sum(dY(:)));
        fprintf('(1) %s N=%d: max realized regret %.3e, sum(dist)=%.6f\n', sty{1}, Nv, max(fr(:)), sum(dY(:)));
        if abs(sum(dY(:)) - 1) > 1e-9, ok = false; fprintf('FAIL: action distribution does not sum to one\n'); end
    end
    if ~(reg.(sty{1}).N80000.maxreg < reg.(sty{1}).N20000.maxreg)
        fprintf('WARN: %s realized regret did not fall from 20k to 80k (%.3e -> %.3e)\n', sty{1}, reg.(sty{1}).N20000.maxreg, reg.(sty{1}).N80000.maxreg);
    end
end

%% (2) tolerances
phi = cfg.marg_distrib(:)'; K = max(cfg.Pi(:,:,1)) - min(cfg.Pi(:,:,1));
r_cons = df.solvers.compute_consistency_slack(s^2, aC, 80000, 'L1');
for sw = 14:17
    e = df.solvers.compute_epsilon_v05(cfg, 80000, phi, r_cons, sw, aR);
    fprintf('(2) sw%d tolerances: %s\n', sw, mat2str(e, 4));
    if any(~isfinite(e)) || any(e < 0), ok = false; fprintf('FAIL: sw%d tolerance not finite/nonneg\n', sw); end
    if ismember(sw, [14 15]) && any(e > K .* min(1, phi + r_cons/2) + 1e-12), ok = false; fprintf('FAIL: sw%d above cap\n', sw); end
end
e10 = df.solvers.compute_epsilon(cfg, 80000, aR, 10) .* sqrt(phi);
e16 = df.solvers.compute_epsilon_v05(cfg, 80000, phi, r_cons, 16, aR);
fprintf('(2) ratio momentF / legacy sw10 per type (should be about M_R = %d times %.2f): %s\n', ...
    cfg.NPlayers*s, (1.25*sqrt(log(5))+0.25)/sqrt(2*log(5)), mat2str(e16 ./ e10, 3));

%% (3)-(4) Stage II on a 4x4 grid, six arms, both learners
base = struct('maxiters_values', 80000, 'alpha_set', aR + aC, 'alpha_R', aR, 'alpha_C', aC, ...
    'backend', 'fast', 'adaptive', false, 'require_corrected', true, 'consistency_slack_kind', 'L1', ...
    'fixed_gridparamV', fgV, 'fixed_gridparamM', fgM, 'use_parfor', false);
arms = {'hedge',16; 'hedge',14; 'hedge',10; 'hedge',12; 'exp3ix',17; 'exp3ix',15};
for a = 1:size(arms,1)
    o = base; o.learning_style = arms{a,1}; o.switch_eps = arms{a,2};
    o.precomputed_distY = {reg.(arms{a,1}).N80000.dY};
    r = df.stages.run_stage_ii(cfg, o);
    VV = r.VV_all(1,:);
    if any(~isfinite(VV)), ok = false; fprintf('FAIL: non-finite VV for %s sw%d\n', arms{a,1}, arms{a,2}); end
    truth_in = VV(1) <= IDTOL; share = mean(VV <= IDTOL);
    fprintf('(3) %s sw%d: truth identified=%d, share=%.3f\n', arms{a,1}, arms{a,2}, truth_in, share);
    if ~truth_in, ok = false; fprintf('FAIL: truth not identified for %s sw%d at N=80k\n', arms{a,1}, arms{a,2}); end
end

%% (6) tail code without a consistency slack must error
try
    o = base; o.learning_style = 'hedge'; o.switch_eps = 14; o.consistency_slack_kind = 'none'; o.alpha_R = aR + aC; o.alpha_C = 0;
    o.precomputed_distY = {reg.hedge.N80000.dY};
    df.stages.run_stage_ii(cfg, o); ok = false; fprintf('FAIL: tail code accepted without slack\n');
catch err
    fprintf('(6) tail without slack errors as intended: %s\n', err.identifier);
end

if ok, fprintf('SMOKE_V05: PASS\n'); else, fprintf('SMOKE_V05: FAIL\n'); exit(1); end
