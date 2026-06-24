%% II_SMOKE_nextwave
%  Fast machinery smoke for the next-wave exercises (D1 fixed-eps, G1 pooled,
%  T1 sharp-volume) and the run_stage_ii driver edits (eps_override, fixed grid,
%  pooled dispatch) + the pooled learner. Tiny T and a 4x4 grid so it finishes in
%  seconds. Validates shapes / integration BEFORE any production cluster run.
%
%  Run via: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_nextwave"
%  Prints SMOKE_OK / SMOKE_FAIL lines and a final ALL_PASS marker.

fprintf('=== NEXTWAVE_SMOKE_START ===\n');
ok = true;

NPlayers = 2; alpha = -(1/3); actions_vec = [4;5;6;7;8];
mu = 3*ones(NPlayers,1); sigma2 = eye(NPlayers); s = 5;
confid = 0.05; switch_eps = 10;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);

NGRID = 4;
fixed_gridparamM = [1; linspace(0.18, 1.5, NGRID)'];
fixed_gridparamV = [1; linspace(0.15, 3.5, NGRID)'];
base = struct('maxiters_values',[20000 40000], 'alpha_set',confid, ...
    'switch_eps',switch_eps, 'backend','fast', 'adaptive',false, ...
    'require_corrected',true, 'fixed_gridparamV',fixed_gridparamV, ...
    'fixed_gridparamM',fixed_gridparamM);

%% (1) D1/T1 path: rm + fixed grid (correct eps), then eps_override + precomputed reuse
try
    o = base; o.learning_style = 'rm';
    rng(12345); res_c = df.stages.run_stage_ii(cfg, o);
    assert(isequal(size(res_c.VV_all), [2, (NGRID+1)^2]), 'VV_all size');
    assert(res_c.NGridV == NGRID && res_c.NGridM == NGRID, 'fixed grid sizes');

    eps_big = df.solvers.compute_epsilon(cfg, 40000, confid, switch_eps);
    assert(isequal(size(eps_big), [1, s]), 'eps_override must be 1xs');
    o2 = o; o2.eps_override = eps_big; o2.precomputed_distY = res_c.distY_time_all;
    rng(12345); res_f = df.stages.run_stage_ii(cfg, o2);
    assert(isequal(size(res_f.VV_all), [2, (NGRID+1)^2]), 'frozen VV_all size');
    assert(~isempty(res_f.eps_override), 'eps_override recorded');
    % frozen eps at smaller N should be no tighter than correct eps -> region no smaller
    a_c = nnz(res_c.VV_all(1,:) <= 1e-8); a_f = nnz(res_f.VV_all(1,:) <= 1e-8);
    fprintf('NEXTWAVE_SMOKE_OK D1/T1: correct nID(N1)=%d, frozen nID(N1)=%d\n', a_c, a_f);
catch e
    ok = false; fprintf('NEXTWAVE_SMOKE_FAIL D1/T1: %s\n', e.message);
end

%% (2) G1 path: pooled learner through the driver
try
    o = base; o.learning_style = 'pooled';
    rng(12345); res_p = df.stages.run_stage_ii(cfg, o);
    assert(isequal(size(res_p.VV_all), [2, (NGRID+1)^2]), 'pooled VV_all size');
    assert(strcmp(res_p.learning_style,'pooled'), 'pooled style recorded');
    fprintf('NEXTWAVE_SMOKE_OK G1 driver: pooled nID(N2)=%d\n', nnz(res_p.VV_all(2,:) <= 1e-8));
catch e
    ok = false; fprintf('NEXTWAVE_SMOKE_FAIL G1 driver: %s\n', e.message);
end

%% (3) Pooled learner direct: regret-unit diagnostic shapes + sanity
try
    T = 40000;
    rng(12345); [dY, ~, fr] = df.sim.learn(cfg, 1, T, T, 1, 1, 1, 1);
    rng(12345); [dYp, ~, frp_pool, frp_type] = df.sim.learn_pooled(cfg, 1, T, T, 1, 1, 1, 1);
    assert(abs(sum(dY(:,1)) - 1) < 1e-9 && abs(sum(dYp(:,1)) - 1) < 1e-9, 'distY sums to 1');
    assert(isequal(size(frp_pool),[1 NPlayers]) && isequal(size(frp_type),[s NPlayers]), 'regret shapes');
    assert(all(isfinite([fr(:); frp_pool(:); frp_type(:)])), 'regrets finite');
    % Defining property: pooled play's type-conditional regret should exceed its
    % pooled regret (it minimizes the pooled notion, not the contextual one).
    fprintf('NEXTWAVE_SMOKE_OK G1 learner: type-play maxTypeReg=%.3e | pooled-play maxPoolReg=%.3e maxTypeReg=%.3e\n', ...
        max(fr(:)), max(frp_pool(:)), max(frp_type(:)));
    if max(frp_type(:)) < max(frp_pool(:))
        fprintf('  NOTE: pooled-play type-regret below pooled-regret at tiny T (may be noise; check at production T)\n');
    end
catch e
    ok = false; fprintf('NEXTWAVE_SMOKE_FAIL G1 learner: %s\n', e.message);
end

fprintf('=== NEXTWAVE_SMOKE_DONE ALL_PASS=%d ===\n', ok);
