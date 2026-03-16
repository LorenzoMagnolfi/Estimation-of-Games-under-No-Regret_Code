# MATLAB Refactor Plan

Last updated: 2026-03-15

## Goal

Refactor the MATLAB layer into a modular, testable pipeline while preserving the current paper's outputs to tight numerical tolerance. Keep the four existing main scripts as thin replication wrappers; move all real work into reusable modules so the revision can reuse the same engine without duplicating logic.

Target outcomes:
- Same baseline outputs, same output locations, same manuscript-facing filenames
- Much faster reruns for the current paper and for revision experiments
- Clear separation between data loading, numerical kernels, solver backends, simulation, and reporting
- No AMPL dependency (replace with native MATLAB `linprog`)

Estimated engineering effort: 15-22 focused working days + 3-5 days validation/benchmarking.

---

## 1. Codebase Architecture Assessment

### Entry points and call graphs

```
I_MAIN_Simul_2acts.m  (Stage I: 2-action polytope + learning demo, ~2 min)
  -> df_repo_paths, marginal_cost_draws_v5, allcomb
  -> find_polytope_switch -> [AMPL external solver — TO BE REPLACED with linprog]
  -> learn (old learning function, NOT learn_mod)
  -> drawBCCE, drawConvergence, drawRegrets_per_period

II_MAIN_simul.m  (Stage II: simulation identification, ~188 min)
  -> df_repo_paths, marginal_cost_draws_v5, allcomb
  -> learn_mod -> marginal_cost_draws_v4_new, regret_matching_mod, choice_prob
  -> ComputeBCCE_eps -> epsilon_switch [uses globals: Pi, NAct, NPlayers, s]
  -> fitcsvm, predict [SVM for identified-set boundary]
  -> [inline plotting + table generation, ~250 lines]

III_MAIN_Estim_Application_PrefSpec.m  (Stage III: empirical application, ~116 min)
  -> df_repo_paths
  -> Identification_Pricing_Game_ApplicationL
     -> get_player_data_5acts, allcomb
     -> ComputeBCCE_eps_ApplicationL -> epsilon_switch_distrib [uses globals]
  -> fitcsvm, predict
  -> get_player_data_5acts, xlsread (repeated)
  -> [inline plotting, table generation, Monte Carlo cost draws, ~250 lines]

IV_MAIN_Emp_Distrib_Regrets.m  (Stage IV: regret distribution + robustness, ~148 min)
  -> df_repo_paths, marginal_cost_draws_v5, allcomb
  -> learn_mod (x500 bootstrap) -> regret_matching_mod, choice_prob
  -> epsilon_switch [direct call for theoretical bounds]
  -> ComputeBCCE_eps_pass -> [takes eps directly, no epsilon_switch call]
  -> fitcsvm, predict, BreakXAxis
  -> [inline plotting, ~150 lines]
```

Total current runtime: ~454 min (~7.6 h).

### The three solver variants

All three solve the same SOCP structure:
```
maximize  -c'x
subject to
  B_EQ * x == beq            (linear equality)
  B_INEQ * x >= b            (linear inequality)
  lb <= x <= ub              (box bounds)
  norm(Mat_NLC * x) <= 1     (second-order cone)
```

What differs:

| Aspect | `ComputeBCCE_eps` | `_ApplicationL` | `_pass` |
|--------|-------------------|-----------------|---------|
| Mode | Joint (2-player) | Marginal (1-player view) | Joint (2-player) |
| BCE measure dim | `dv = s^2 * NA` | `dv = s * Nactions` | `dv = s^2 * NA` |
| Epsilon source | `epsilon_switch()` | `epsilon_switch_distrib()` | Direct argument |
| Equality constraints | `deq = 1 + NA + s^2` | `deq = 1 + Nactions + s` | `deq = 1 + NA + s^2` |
| Output shape | `g(NAlpha, NGrid, Nq_N)` | `g(NGrid, Nq_N)` | `g(NAlpha, NGrid, Nq_N)` |
| Alpha loop | Yes | No | Yes |
| CVX solver spec | Default (SDPT3) | `cvx_solver sedumi` | Default |
| Data preprocessing | None | Marginalizes joint action distrib | None |

`ComputeBCCE_eps` and `_pass` are structurally identical; only the epsilon source differs. `_ApplicationL` has genuinely different problem dimensions (marginal vs joint) — this is a real conditional branch, not just a thin adapter.

### Critical bottleneck insight

The constraint matrices (`B_EQ`, `B_INEQ`, bounds, `Mat_NLC`) are **invariant across the inner parameter-grid loops**. Only the objective vector `c` changes per grid point. Yet all three variants rebuild these matrices inside the loop. For a 100x100 grid, that is 10,000 redundant matrix constructions per solver call.

Worse, each of those 10,000 iterations invokes the full CVX modeling layer (parse, transform, dispatch), adding ~0.1-0.5s of overhead per solve on top of the ~0.01-0.05s actual SOCP solve time. **CVX parsing overhead dominates wall-clock time in Stages II-IV.**

### The epsilon function problem

Two epsilon functions exist with overlapping but inconsistent formulas:

| `switch_eps` | `epsilon_switch.m` (Stages I, II, IV) | `epsilon_switch_distrib.m` (Stage III) |
|:---:|---|---|
| 1 | `Kappa * sqrt(log(NAct)) / (conf * sqrt(T))` | `s * Kappa * sqrt(log(NAct)) / (conf * sqrt(T))` |
| 3 | `Kappa * conf * sqrt(log(NAct))` | `Kappa * conf * sqrt(log(NAct))` (same) |
| 5 | `(NAct-1) / (Kappa * conf * T)` | `TType_spec / (conf * T)` (different formula) |
| 6-9 | Not defined | Stochastic bandit bounds with gap-dependent terms |

The `_distrib` version uses opponent-marginal expected payoffs to compute gap-dependent bounds (`TType_spec_LargestDev2`), which is specific to the application setting. The unified epsilon interface must preserve which formula each stage actually uses.

### Global variable dependency

12+ variables passed via `global` across the codebase:

| Variable | Set by | Read by |
|----------|--------|---------|
| `NAct`, `NPlayers`, `s` | MAIN scripts | `epsilon_switch`, `epsilon_switch_distrib` |
| `Pi` | MAIN scripts | `epsilon_switch`, `epsilon_switch_distrib` |
| `alpha` | MAIN scripts | `learn_mod` (via `choice_prob`) |
| `A`, `AA` | MAIN scripts | `learn_mod`, `regret_matching_mod` |
| `type_space` | MAIN scripts | `learn_mod` |
| `learning_style` | MAIN scripts | `learn_mod` |
| `mu`, `sigma2`, `Egrid`, `Psi`, `marg_distrib`, `tps` | MAIN scripts | Various (some unused in callees) |

This makes every function implicitly coupled to every MAIN script. Must be eliminated before any modularization.

### Game setup duplication

All four MAIN scripts and `Identification_Pricing_Game_ApplicationL.m` duplicate 40-60 lines of identical logic:
- Type space construction via `marginal_cost_draws_v5`
- Joint type profiles via Kronecker products (`T_sorted`)
- Prior `Psi` from product of marginals
- Utility tensor `Pi` from logit choice probabilities times markup

---

## 2. Baseline Capture Strategy

### What the paper/replication package already provides

The existing replication package and paper outputs serve as the **final artifact oracle**:
- All published figures (`.eps`, `.png`)
- All published tables (`.tex`, `.xlsx`)
- Stage IV `.mat` files (`final_regret.mat`, `ratio*.mat`)
- README-documented runtimes for the original machine

These suffice for verifying that refactored code produces the same manuscript outputs. **A full 7.6-hour rerun is not needed for this purpose.**

### What is missing (and why a targeted capture run is needed)

The current scripts do NOT persist key intermediate arrays that are needed for **module-level regression testing**:

| Stage | Missing intermediates | Why needed |
|-------|----------------------|------------|
| II | `VV` (objective values), `id_set_index` (membership), `distY_time` (learned distributions) | Validate solver kernel independently of SVM/plotting |
| III | `maxvals` (raw solver output per player), `ddpars` (parameter grid) | Validate application solver path |
| IV | `avg_exp_regret2` (bootstrap regret averages), `VV` (identification objective) | Validate bootstrap + pass-through solver |
| I | Polytope vertices from `find_polytope_switch` | Validate `linprog` replacement produces same vertices |

### Recommended approach: instrumented partial runs

Instead of a gated "Phase 0" requiring the full 7.6h, do two lightweight steps:

**Step A** (~10 min of editing, no MATLAB run): Add `save` calls to each MAIN at the points where key intermediates are computed. Example for `II_MAIN_simul.m`:
```matlab
% After line 231 (solver output):
save(fullfile(paths.artifacts_ii, sprintf('baseline_VV_%dk.mat', maxiters/1000)), 'VV', 'id_set_index', 'distpars');
```

**Step B** (partial runs, ~1-2h total instead of 7.6h): Run each stage with **reduced problem size** on the actual machine to capture:
- Module-level intermediates (solver `g`, learning `distY_time`, epsilon values)
- Per-phase wall-clock times
- Solver call counts

Reduced-size configuration for fixture capture:
| Stage | Full config | Fixture config | Est. time |
|-------|------------|----------------|-----------|
| I | `s=20, maxiters=10k` | Same (already fast) | 2 min |
| II | `s=5, 100x100 grid, 4 maxiters values` | `s=5, 20x20 grid, 1 maxiters value (500k)` | ~8 min |
| III | `100x100 grid, 2 players` | `20x20 grid, 1 player` | ~5 min |
| IV | `B=500 bootstrap, 100x100 grid` | `B=10 bootstrap, 20x20 grid` | ~10 min |

Store fixture outputs under `matlab/test/fixtures/` with the reduced configs documented.

**Step C** (one full run, timed, on the real machine when convenient): Run the unmodified scripts at full scale to record production timings. This is not a blocker for starting refactor work — it can happen in parallel with Phase 1.

### Validation script

Write `matlab/test/verify_refactor.m` that:
1. Loads fixture `.mat` files (reduced-size intermediates)
2. Runs the refactored pipeline with the same reduced-size config and same RNG seeds
3. Compares solver objective values to tolerance 1e-10
4. Compares identified-set membership exactly (binary match)
5. Compares learning distributions to tolerance 1e-12
6. Reports PASS/FAIL per stage

For final manuscript validation, visually compare generated figures to paper originals and diff `.tex` tables.

---

## 3. Module Layout

Adopt MATLAB package namespaces (`+df/`) for clean organization:

```
matlab/src/
├── +df/
│   ├── +io/
│   │   ├── load_workbook_cached.m    % parse xlsx/csv once, cache as .mat
│   │   ├── get_player_data.m         % wraps get_player_data_5acts with caching
│   │   └── repo_paths.m             % replaces df_repo_paths
│   ├── +setup/
│   │   ├── game_simulation.m        % setup for Stages I, II, IV (simulation params)
│   │   └── game_application.m       % setup for Stage III (data-driven params)
│   ├── +solvers/
│   │   ├── solve_bcce.m             % unified solver entry point
│   │   ├── build_constraints.m      % equality, inequality, bounds, cone — built ONCE
│   │   ├── build_constraints_marginal.m  % marginal-mode variant (different dimensions)
│   │   ├── compute_epsilon.m        % unified epsilon dispatcher
│   │   ├── solve_socp_coneprog.m    % primary backend (R2020b+)
│   │   ├── solve_socp_cvx.m         % validation/fallback backend
│   │   └── solve_polytope_lp.m      % replaces find_polytope_switch + AMPL
│   ├── +sim/
│   │   ├── learn.m                  % refactored learning kernel
│   │   ├── regret_matching.m        % regret matching decision rule
│   │   ├── choice_prob.m            % logit choice probabilities
│   │   └── marginal_cost_draws.m    % type-space draws (merges v4/v4_new/v5)
│   ├── +stages/
│   │   ├── run_stage_i.m            % polytope + learning demo
│   │   ├── run_stage_ii.m           % simulation identification
│   │   ├── run_stage_iii.m          % application estimation
│   │   └── run_stage_iv.m           % regret distribution + robustness
│   ├── +report/
│   │   ├── plot_identified_set.m    % SVM boundary + projections (shared II, III, IV)
│   │   ├── plot_convergence.m       % learning convergence (Stage I)
│   │   ├── plot_regret_distribution.m  % regret histogram (Stage IV)
│   │   ├── classify_identified_set.m   % SVM wrapper (shared II, III, IV)
│   │   ├── build_param_grid.m       % parameter grid construction (shared)
│   │   └── write_tables.m           % xlsx + LaTeX table export
│   └── +util/
│       ├── allcomb.m                % all combinations (existing)
│       └── table2latex.m            % existing utility
├── df_run.m                         % public API: df_run('ii', opts)
├── I_MAIN_Simul_2acts.m             % thin wrapper -> df_run('i', ...)
├── II_MAIN_simul.m                  % thin wrapper -> df_run('ii', ...)
├── III_MAIN_Estim_Application_PrefSpec.m  % thin wrapper -> df_run('iii', ...)
├── IV_MAIN_Emp_Distrib_Regrets.m    % thin wrapper -> df_run('iv', ...)
└── test/
    ├── verify_refactor.m            % regression test runner
    └── fixtures/                    % saved baseline intermediates
```

### `df_run` public API

```matlab
function results = df_run(stage, opts)
% stage: 'i', 'ii', 'iii', 'iv'
% opts (struct, all optional with defaults):
%   .backend       'coneprog' (default) | 'cvx'
%   .parallel      'auto' (default: use PCT if available) | 'off'
%   .rng_seed      stage-specific default matching current scripts
%   .save_outputs  true (default) | false
%   .save_intermediates  false (default) | true (for fixture capture)
%   .grid_size     [NGridM, NGridV] override for reduced runs
%   .bootstrap_B   override for Stage IV bootstrap count
```

---

## 4. Phased Implementation

### Phase 1: Eliminate Globals + Extract Game Setup

**Combined because they're tightly coupled**: you can't extract setup into a function while the function's callees read globals.

**Deliverable**: A `cfg` struct replaces all global state.

```matlab
cfg = struct();
cfg.NPlayers       = 2;
cfg.alpha          = -(1/3);
cfg.NAct           = 5;
cfg.A              = allcomb(...);   % action profiles matrix
cfg.AA             = [4;5;6;7;8];   % individual actions vector
cfg.s              = 5;             % number of types per player
cfg.type_space     = {...};         % cell array of type vectors
cfg.marg_distrib   = [...];         % marginal prior over types
cfg.Pi             = [...];         % utility tensor (NActPr x s x NPlayers)
cfg.Psi            = [...];         % joint prior over type profiles
cfg.mu             = [...];         % mean MC parameter
cfg.sigma2         = [...];         % variance MC parameter
cfg.learning_style = 'rm';
```

**Tasks**:
1. Create `+df/+setup/game_simulation.m`: takes `(NPlayers, alpha, actions, mu, sigma2, s)`, returns `cfg`
2. Create `+df/+setup/game_application.m`: takes `(player_id, Dist_file, Prob_file, n_types)`, returns `cfg` with data-driven actions and sale probabilities
3. Modify every function to accept `cfg` as explicit argument:
   - `learn_mod(cfg, N, M, ...)` instead of reading `A, AA, type_space, learning_style, alpha` from globals
   - `epsilon_switch(cfg, maxiters, conf, switch_eps)` instead of reading `Pi, NAct, NPlayers, s` from globals
   - Same for `regret_matching_mod`, `choice_prob`, all three `ComputeBCCE_eps*`
4. Remove all `global` declarations
5. Replace inline setup blocks in all four MAINs with `cfg = df.setup.game_simulation(...)` or `cfg = df.setup.game_application(...)`
6. Validate: run with fixtures, compare outputs

**Files touched**: All `.m` files in `matlab/src/` (every function and every MAIN).

**Risk**: Low. Mechanical transformation, no computation change.

### Phase 2: Unify Solver Kernel

**Deliverable**: Single `df.solvers.solve_bcce(cfg, solver_opts)` replaces three functions.

**Internal design**:
```
solve_bcce(cfg, solver_opts)
  │
  ├─ if solver_opts.marginal_mode
  │    constraints = build_constraints_marginal(cfg, solver_opts)
  │  else
  │    constraints = build_constraints(cfg, solver_opts)    ← BUILT ONCE
  │
  ├─ eps_vec = compute_epsilon(cfg, solver_opts)            ← unified dispatcher
  │    handles: switch_eps 0-9, eps_override for pass-through
  │
  ├─ for each (nd, nb) grid point:
  │    c = build_objective(constraints, eps_vec, nd, nb, ...)  ← only c changes
  │    g(nd,nb) = solve_single_socp(constraints, c, backend)
  │
  └─ return results struct with g, solve_time, n_solves
```

**Key moves**:
1. **Hoist constraint construction**: `M`, `M_prime`, `B_EQ`, `B_INEQ` are currently rebuilt inside the `for nb` loop in all three variants. They don't depend on `nb` (or on `nd` or `np`). Build once, reuse.
2. **Unify epsilon interface**: `compute_epsilon(cfg, T, confid, switch_eps, varargin)` dispatches to the right formula. Accepts optional `eps_override` (for pass-through mode) and `marg_mean` (for application mode with gap-dependent bounds).
3. **Joint vs marginal as a flag**: The marginal mode (ApplicationL) uses `Nactions` instead of `NA` for constraint dimensions, marginalizes the action distribution, computes expected payoffs via `kron(eye(Nactions), marg_act_distrib') * Pi`. This is a **conditional branch in the constraint builder**, not a separate function — the actual SOCP solve is identical.
4. **Absorb `Identification_Pricing_Game_ApplicationL.m`**: Its setup logic moves to `df.setup.game_application`; its solver call becomes `df.solvers.solve_bcce` with `marginal_mode=true`.

**Epsilon reconciliation detail**:
```matlab
function eps = compute_epsilon(cfg, T, confid, switch_eps, kwargs)
% kwargs.eps_override:  if provided, return directly (pass-through mode)
% kwargs.marg_mean:     required for switch_eps >= 6 (gap-dependent bounds)
%
% switch_eps mapping:
%   0:   fixed eps, new largest deviation
%   1:   convergence-rate bound (NOTE: simulation uses Kappa*..., application uses s*Kappa*...)
%   2:   fixed eps, old largest deviation (2 actions only)
%   3:   fixed eps, Feb 2024 definition
%   4:   EXP3 bounds
%   5:   stochastic bandit, conservative
%   6-9: stochastic bandit with gap-dependent terms (application only)
```

The `s` multiplier discrepancy for `switch_eps==1` between `epsilon_switch` and `epsilon_switch_distrib` must be preserved as-is (Stage II uses the version without `s`; Stage III uses the version with `s`). Document this in the function.

**Files created**: `+df/+solvers/solve_bcce.m`, `build_constraints.m`, `build_constraints_marginal.m`, `compute_epsilon.m`, `solve_socp_cvx.m`
**Files deprecated**: `ComputeBCCE_eps.m`, `ComputeBCCE_eps_ApplicationL.m`, `ComputeBCCE_eps_pass.m`, `epsilon_switch.m`, `epsilon_switch_distrib.m`, `Identification_Pricing_Game_ApplicationL.m`

### Phase 3: Add `coneprog` Backend + Kill AMPL

**3a: `coneprog` for SOCP (Stages II-IV)**

Mapping to `coneprog` interface (requires MATLAB R2020b+):
```matlab
% coneprog: min f'x  s.t.  Aineq*x <= bineq, Aeq*x = beq, lb <= x <= ub, cone constraints

f     = c;                              % objective (minimize c'x = maximize -c'x)
Aeq   = B_EQ;    beq_val = beq;        % equality
Aineq = -B_INEQ; bineq   = -b;         % B_INEQ*x >= b  <=>  -B_INEQ*x <= -b
% lb, ub as-is
socConstraint = secondordercone(Mat_NLC, zeros(NA-1,1), zeros(1,n), 1);
%   meaning: ||Mat_NLC * x + 0|| <= 0'x + 1, i.e., ||Mat_NLC * x|| <= 1

[x, fval, exitflag] = coneprog(f, socConstraint, Aineq, bineq, Aeq, beq_val, lb, ub);
```

Gate with version check:
```matlab
if exist('coneprog', 'file') && ~strcmp(solver_opts.backend, 'cvx')
    % use coneprog
else
    % fall back to CVX
end
```

**Expected speedup**: Eliminating CVX parsing overhead (~0.1-0.5s per solve) is the dominant gain. For 10,000 grid-point solves, this saves ~1,000-5,000s. The actual SOCP solve time (~0.01-0.05s per problem) is unchanged. Net serial speedup on the solver phase: **~2x-5x**, depending on how much time was CVX overhead vs actual solve.

**3b: `linprog` for polytope (Stage I)**

`find_polytope_switch.m` currently writes AMPL `.dat`/`.run` files and shells out to an AMPL executable. The underlying problem is a **linear program** for polytope vertex enumeration — not an SOCP.

Replace with `linprog`:
```matlab
% Current AMPL call solves: max/min u'q  s.t.  constraint set
% This maps directly to:
[x, fval] = linprog(-u, A_ineq, b_ineq, A_eq, b_eq, lb, ub);
```

**Tasks**:
1. Read `find_polytope_switch.m` and the `.mod`/`.dat` files to extract the LP formulation
2. Implement `df.solvers.solve_polytope_lp.m` using `linprog`
3. Validate: compare polytope vertices to AMPL output on Stage I fixture
4. Remove AMPL files from `matlab/src/` (`.mod`, `.dat`, `.run`)
5. Remove AMPL path resolution from `df_repo_paths.m`

**Risk**: Low. LPs are well-handled by `linprog`. The AMPL problem is small.

### Phase 4: Optimize Learning Kernel

**Deliverable**: Faster `df.sim.learn.m` with identical statistical behavior.

**Current bottleneck in `learn_mod.m`**: The main loop runs `N+M` iterations (up to 4M for Stage II). Each iteration does:
- Type draw + index lookup: `find(type_space{j,1} == mc_draw(j))` — linear search with exact float equality
- Per-player, per-type, per-action updates: `O(NPlayers * s * |AA|)` per iteration
- Action profile lookup: `find(all(A == actions, 2))` — linear scan every iteration
- Distribution recording: `ismember(t, round(...))` — evaluated every iteration

**Semantic caveat on the decay rule**: The current code (lines 54-71) decays ALL types' running averages every period, not just the realized one:
```matlab
if tt == type_j
    U_n(tt, j) = ((t-1) * U_n(tt, j) + utility) / t;
else
    U_n(tt, j) = ((t-1) * U_n(tt, j)) / t;
end
```
This means `U_n(tt,j) = cumulative_sum(tt,j) / t`, where the cumulative sum only increments when type `tt` is actually drawn. This is correct: regret is measured per unit of calendar time, not per type-realization. The sufficient-statistics rewrite must preserve this: store `cum_util(tt,j)` and `cum_cf_util(a,tt,j)`, derive averages as `cum/t` only when needed for regret matching.

**Optimizations**:
1. **Precompute recording times**: `record_times = round(M * (1:numdst_t)/numdst_t)` before the loop; check with `t == record_times(next_record_idx)`
2. **Replace `find` lookups with index tables**: Precompute `type_to_idx` map; precompute `action_profile_to_idx` as a 2D lookup array
3. **Track cumulative sums, derive averages on demand**: Store `cum_util(tt,j)` and `cum_cf_util(a,tt,j)`. Only compute `U_n = cum_util / t` when calling `regret_matching_mod`. Eliminates the `s * NPlayers` decay multiplications per iteration for unrealized types.
4. **Vectorize counterfactual utility computation**: Replace per-action `choice_prob` calls with a vectorized batch over all actions
5. **Merge marginal cost draws**: Unify `marginal_cost_draws_v4.m`, `_v4_new.m`, `_v5.m` into `df.sim.marginal_cost_draws.m`

**Parallelization for Stage IV bootstrap**: The 500-iteration bootstrap in `IV_MAIN` is embarrassingly parallel. Use `parfor` with `RandStream` substreams:
```matlab
streams = RandStream.create('mrg32k3a', 'NumStreams', B, 'Seed', opts.rng_seed);
parfor b = 1:B
    stream = streams{b};
    % pass stream to learn() for all random draws
    [distY, ~, fin, ...] = df.sim.learn(cfg, N, M, ..., 'rng_stream', stream);
    final_regret(:,:,b) = fin;
end
```

**Expected speedup**: `learn_mod` serial: 3x-5x from eliminating redundant decay updates and vectorizing inner loops. Stage IV with `parfor`: additional 3x-5x depending on core count.

### Phase 5: Separate Compute from Reporting

**Deliverable**: Each stage produces a `results` struct; plotting/export is a separate step.

**Target architecture**:
```matlab
% In df_run('ii', opts):
cfg = df.setup.game_simulation(params);
results = df.stages.run_stage_ii(cfg, opts);   % learning + solver + SVM
if opts.save_outputs
    df.report.plot_identified_set(results, paths.figures_ii);
    df.report.write_tables(results, paths.tables_ii);
end
```

**Shared reporting modules**:
- `plot_identified_set.m`: SVM boundary visualization with mu/sigma projections (shared across Stages II, III, IV)
- `classify_identified_set.m`: `fitcsvm` + `predict` on Halton grid (shared)
- `build_param_grid.m`: Construct `(gridparamM, gridparamV, distribution_parameters)` from config (shared, eliminates inline grid construction in each MAIN)
- `write_tables.m`: `.xlsx` + `.tex` export with consistent formatting

**Stage III specific**: The Monte Carlo cost-draw section (lines 148-214 of `III_MAIN`) that samples marginal costs from the identified set and computes markup statistics becomes `df.stages.compute_cost_statistics.m`. The Gazelle/Swappa data merge (lines 294-347) becomes part of `df.report.write_tables.m`.

---

## 5. Performance Targets

Current baseline (from replication package README):

| Stage | Current | Primary bottleneck |
|-------|--------:|-------------------|
| I | 2 min | AMPL LP solve + learning |
| II | 188 min | 40,000 CVX solves + 4x learning |
| III | 116 min | ~20,000 CVX solves (2 players x 10k grid) |
| IV | 148 min | 500x bootstrap learning + 10,000 CVX solves |
| **Total** | **454 min** | |

Refactor targets (same machine):

| Stage | Target Serial | Target Parallel | Main driver |
|-------|-------------:|----------------:|-------------|
| I | 1-2 min | 1 min | `linprog` replacing AMPL |
| II | 70-110 min | 25-45 min | `coneprog` + hoisted constraints + parallel grid |
| III | 40-70 min | 15-30 min | `coneprog` + cached I/O + shared solver |
| IV | 60-90 min | 20-35 min | `learn` rewrite + `parfor` bootstrap + `coneprog` |
| **Total** | **171-272 min** | **61-111 min** | |

Conservative serial target: ~2x-2.7x. Ambitious parallel target: ~4x-7x.

**Caveat on serial speedup estimates**: The solver speedup is almost entirely from eliminating CVX parsing overhead, not from sparse reuse (the constraint matrices are dense Kronecker products at this problem size). The "2x-5x solver speedup" range depends on the fraction of wall-clock time that is CVX overhead vs actual SOCP solve. This will only be known precisely after the timed baseline run.

---

## 6. Dependency Order and Critical Path

```
Baseline Capture (Steps A+B: instrument + partial runs)
    |
Phase 1: Globals + Game Setup
    |
Phase 2: Unified Solver Kernel
    |
    ├── Phase 3a: coneprog backend    ← can start immediately after Phase 2
    ├── Phase 3b: Kill AMPL (linprog) ← independent, can start after Phase 1
    └── Phase 4: learn_mod rewrite    ← independent, can start after Phase 1
    |
Phase 5: Compute/Plot separation      ← after Phase 2
```

Phases 3a, 3b, 4 are independent of each other and can proceed in parallel.
Phase 5 depends on Phase 2 (needs the unified solver interface to define the results struct).
Baseline capture is a lightweight prep step, not a blocking gate.

---

## 7. Risk Assessment

| Phase | Risk | Mitigation |
|-------|------|------------|
| Baseline | Can't run MATLAB on this machine | Run on lab machine; partial-run fixtures are small (~10 min of editing) |
| 1 | Mechanical but touches every file | Automated grep for `global` declarations; one-function-at-a-time migration |
| 2 | Joint vs marginal constraint dims differ | Explicit `build_constraints_marginal.m`, not a "thin adapter" pretending dims are the same |
| 2 | Epsilon formula discrepancy (s multiplier) | Document per-stage behavior in `compute_epsilon`; preserve existing formulas exactly |
| 3a | `coneprog` unavailable in older MATLAB | Runtime version check; CVX fallback always available |
| 3b | AMPL LP may have subtleties not visible in `.mod` file | Compare polytope vertices to AMPL output before removing AMPL |
| 4 | RNG-sensitive learning; `parfor` changes iteration order | Use `RandStream` substreams; serial path must remain for exact reproducibility |
| 4 | `learn` (old) vs `learn_mod` may not be equivalent | Stage I keeps `learn` as-is; unification is out of scope |
| 5 | Low risk, mostly code motion | Visual diff of generated figures; diff `.tex` tables |

---

## 8. Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-03-15 | Keep MATLAB, no language migration | CVX/coneprog ecosystem, validated results, no benefit to porting |
| 2026-03-15 | Kill AMPL dependency entirely | Replace Stage I LP with native `linprog`; eliminates external binary |
| 2026-03-15 | Paper outputs serve as final artifact oracle | No full rerun needed; targeted fixture capture for module-level validation |
| 2026-03-15 | Use `+df/` package namespace | Cleaner than flat files; avoids name collisions; MATLAB-idiomatic |
| 2026-03-15 | Three solver variants, not two | `ComputeBCCE_eps_pass.m` was missed in earlier plans; must be included |
| 2026-03-15 | Joint vs marginal is a real conditional branch | `_ApplicationL` has different constraint dimensions; not reducible to a thin adapter |
| 2026-03-15 | Epsilon formula discrepancy is preserved, not "fixed" | `switch_eps==1` intentionally differs between simulation and application stages |
| 2026-03-15 | `I_MAIN` / `learn` (old) left as-is for now | Different code path; refactoring provides little reuse; revisit in revision track |
| 2026-03-15 | Effort estimate: 15-22 days code + 3-5 days validation | Accounts for three solver variants, epsilon complexity, RNG-sensitive learning |

---

## 9. Implementation Roadmap

Concrete work items organized into sessions. Each session is a self-contained unit of work that leaves the codebase in a runnable state. Estimated effort per session assumes a focused working block (3-5 hours).

### Prep: Baseline Fixture Capture (1 session)

**Prerequisites**: MATLAB machine with CVX + Statistics/ML Toolbox.

| # | Task | Files | Deliverable |
|---|------|-------|-------------|
| P.1 | Add `save` calls after key intermediates in all four MAINs | `II_MAIN_simul.m`, `III_MAIN_Estim_Application_PrefSpec.m`, `IV_MAIN_Emp_Distrib_Regrets.m`, `I_MAIN_Simul_2acts.m` | Instrumented scripts |
| P.2 | Create reduced-config fixture runner that calls each MAIN with small grids | New: `matlab/test/run_fixtures.m` | Runner script |
| P.3 | Run fixture runner, commit outputs | New: `matlab/test/fixtures/*.mat` | Regression oracle |
| P.4 | Record per-stage wall-clock times in fixture config | New: `matlab/test/fixtures/timing_baseline.md` | Timing reference |

### Phase 1, Session 1: Package Skeleton + Config Struct (1 session)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 1.1 | Create `+df/` package directory tree | `+df/+io/`, `+df/+setup/`, `+df/+solvers/`, `+df/+sim/`, `+df/+stages/`, `+df/+report/`, `+df/+util/` | Empty dirs + placeholder `Contents.m` |
| 1.2 | Create `df.io.repo_paths()` from `df_repo_paths.m` | New: `+df/+io/repo_paths.m` | Same logic, package namespace |
| 1.3 | Move `allcomb.m`, `table2latex.m` into `+df/+util/` | Move + update callers | |
| 1.4 | Define the `cfg` struct schema | New: `+df/+setup/game_simulation.m` (stub with struct definition) | Document every field |

### Phase 1, Session 2: De-Global the Solver Chain (1-2 sessions)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 1.5 | Refactor `epsilon_switch.m` to accept `cfg` | `epsilon_switch.m` → `+df/+solvers/compute_epsilon.m` | Remove `global Pi NAct NPlayers s`; accept them via cfg |
| 1.6 | Refactor `epsilon_switch_distrib.m` into the same function | `epsilon_switch_distrib.m` → merge into `compute_epsilon.m` | Add `mode` parameter: `'simulation'` vs `'application'`; preserve formula differences |
| 1.7 | Refactor `ComputeBCCE_eps.m` to accept `cfg` | Keep in place, add `cfg` arg, remove globals | Intermediate step; will be replaced in Phase 2 |
| 1.8 | Refactor `ComputeBCCE_eps_ApplicationL.m` to accept `cfg` | Same | |
| 1.9 | Refactor `ComputeBCCE_eps_pass.m` to accept `cfg` | Same | |
| 1.10 | Refactor `choice_prob.m` — no globals needed (already clean) | Verify, move to `+df/+sim/` | |
| 1.11 | Refactor `regret_matching_mod.m` to accept action set as arg | Remove implicit `AA` dependency | |
| 1.12 | Validate: run fixtures, compare to saved outputs | `matlab/test/verify_refactor.m` | |

### Phase 1, Session 3: De-Global the Learning Chain (1 session)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 1.13 | Refactor `learn_mod.m` to accept `cfg` | Remove `global A AA NPlayers alpha type_space learning_style` | Pass all through cfg |
| 1.14 | Refactor `learn.m` (Stage I old learning) to accept `cfg` | Same treatment | Stage I needs this |
| 1.15 | Refactor `marginal_cost_draws_v4_new.m` to accept type_space arg | Remove implicit dependency | |
| 1.16 | Complete `df.setup.game_simulation()` | Pull shared setup code from `II_MAIN_simul.m` | type_space, Psi, Pi construction |
| 1.17 | Create `df.setup.game_application()` | Pull from `Identification_Pricing_Game_ApplicationL.m` | Data-driven setup |
| 1.18 | Update all four MAINs to use `cfg = df.setup.game_simulation(...)` | Remove inline setup blocks | |
| 1.19 | Remove all `global` declarations from all files | Grep-verify: zero hits for `^global` | |
| 1.20 | Validate: run fixtures | | **Phase 1 gate: all fixtures pass with zero globals** |

### Phase 2, Session 1: Constraint Builder (1-2 sessions)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 2.1 | Extract shared constraint construction from `ComputeBCCE_eps.m` | New: `+df/+solvers/build_constraints.m` | Equality, inequality, bounds, cone — returns a `constraints` struct |
| 2.2 | Extract marginal-mode constraint construction from `ComputeBCCE_eps_ApplicationL.m` | New: `+df/+solvers/build_constraints_marginal.m` | Different dimensions; not a trivial flag |
| 2.3 | Verify constraint matrices match originals | Unit test: build with new function, compare to inline construction | |
| 2.4 | Extract CVX solve block into standalone function | New: `+df/+solvers/solve_socp_cvx.m` | Input: `constraints` struct + objective `c`; output: `optval`, `status` |

### Phase 2, Session 2: Unified Solver (1-2 sessions)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 2.5 | Create `df.solvers.solve_bcce()` | New: `+df/+solvers/solve_bcce.m` | Orchestrates: build constraints once, loop over grid, call solver per point |
| 2.6 | Handle the three epsilon modes inside `solve_bcce` | Uses `compute_epsilon` with mode dispatch | `switch_eps` formula, `eps_override` for pass-through |
| 2.7 | Handle joint vs marginal via `solver_opts.marginal_mode` | Dispatches to `build_constraints` vs `build_constraints_marginal` | |
| 2.8 | Replace `ComputeBCCE_eps` calls in `II_MAIN_simul.m` | `outs = df.solvers.solve_bcce(cfg, solver_opts)` | |
| 2.9 | Replace `ComputeBCCE_eps_ApplicationL` calls in `III_MAIN` (via `Identification_Pricing_Game_ApplicationL.m`) | Same | |
| 2.10 | Replace `ComputeBCCE_eps_pass` calls in `IV_MAIN` | Same, with `solver_opts.eps_override` | |
| 2.11 | Validate: all fixtures pass | | **Phase 2 gate: single solver, three old files deprecated** |

### Phase 3a: coneprog Backend (1 session)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 3a.1 | Implement `df.solvers.solve_socp_coneprog()` | New: `+df/+solvers/solve_socp_coneprog.m` | `coneprog` + `secondordercone` mapping |
| 3a.2 | Add backend dispatch to `solve_bcce` | Modify `solve_bcce.m` | `if strcmp(opts.backend, 'coneprog') ... else ... cvx` |
| 3a.3 | Cross-validate: run fixtures with both backends | Compare `g` values to 1e-8 tolerance | |
| 3a.4 | Benchmark: per-solve timing comparison | Record in `matlab/test/fixtures/backend_benchmark.md` | **Phase 3a gate: coneprog matches CVX to tolerance** |

### Phase 3b: Kill AMPL (1 session)

| # | Task | Files | Notes |
|---|------|-------|-------|
| 3b.1 | Read `find_polytope_switch.m` and `Polytope_Pricing.mod` to extract LP formulation | Research only | Document the LP constraints and objective |
| 3b.2 | Implement `df.solvers.solve_polytope_lp()` using `linprog` | New: `+df/+solvers/solve_polytope_lp.m` | |
| 3b.3 | Update `I_MAIN_Simul_2acts.m` to call new function | Replace `find_polytope_switch` call | |
| 3b.4 | Validate: compare polytope vertices to AMPL baseline | Stage I fixture | |
| 3b.5 | Remove AMPL files | Delete: `Polytope_Pricing.mod`, `.dat`, `_new.mod`, `.run` files | |
| 3b.6 | Remove AMPL path resolution from `df_repo_paths.m` / `df.io.repo_paths` | | **Phase 3b gate: Stage I runs with no AMPL** |

### Phase 4: Learning Kernel Rewrite (2-3 sessions)

**Session 1: Core rewrite**

| # | Task | Files | Notes |
|---|------|-------|-------|
| 4.1 | Create `df.sim.learn()` with sufficient-statistics state | New: `+df/+sim/learn.m` | `cum_util`, `cum_cf_util` arrays; derive averages on demand |
| 4.2 | Precompute recording times | Replace `ismember` with indexed check | |
| 4.3 | Precompute action-profile index lookup table | `action_idx_map` array; replace `find(all(A == actions, 2))` | |
| 4.4 | Precompute type index lookup | `type_idx_map`; replace `find(type_space{j,1} == mc_draw(j))` with tolerance-based lookup | |
| 4.5 | Vectorize counterfactual utility computation | Batch `choice_prob` over all actions at once | |
| 4.6 | Validate: compare `distY_time`, `final_regret` to fixtures with same RNG seed | | Must match to 1e-12 |

**Session 2: Marginal cost draws unification + Stage I learning**

| # | Task | Files | Notes |
|---|------|-------|-------|
| 4.7 | Merge `marginal_cost_draws_v4.m`, `_v4_new.m`, `_v5.m` into `df.sim.marginal_cost_draws()` | New: `+df/+sim/marginal_cost_draws.m` | Version dispatch via parameter |
| 4.8 | Refactor `learn.m` (old, Stage I) similarly if feasible | `+df/+sim/learn_legacy.m` or fold into `df.sim.learn` | Stage I uses different regret tracking |
| 4.9 | Validate Stage I fixture | | |

**Session 3: Parallelization (Stage IV)**

| # | Task | Files | Notes |
|---|------|-------|-------|
| 4.10 | Add `RandStream` substream support to `df.sim.learn()` | Optional `rng_stream` argument | |
| 4.11 | Implement `parfor` bootstrap in `df.stages.run_stage_iv()` | `parfor` with `Substream` | |
| 4.12 | Validate: serial path produces same results as before | Fixture comparison | |
| 4.13 | Benchmark: serial vs parallel on Stage IV fixture | Record speedup | **Phase 4 gate: learning kernel passes all fixtures, parallel works** |

### Phase 5: Compute/Plot Separation (2 sessions)

**Session 1: Shared reporting modules**

| # | Task | Files | Notes |
|---|------|-------|-------|
| 5.1 | Create `df.report.classify_identified_set()` | New: extract SVM logic from all MAINs | `fitcsvm` + `predict` on Halton grid |
| 5.2 | Create `df.report.plot_identified_set()` | New: extract plotting from II, III, IV | SVM boundary + mu/sigma projections |
| 5.3 | Create `df.report.build_param_grid()` | New: extract grid construction | `(gridparamM, gridparamV, distribution_parameters)` |
| 5.4 | Create `df.report.write_tables()` | New: `.xlsx` + `.tex` export | |

**Session 2: Stage orchestrators + df_run**

| # | Task | Files | Notes |
|---|------|-------|-------|
| 5.5 | Create `df.stages.run_stage_i()` through `run_stage_iv()` | New: stage orchestrators | Setup -> compute -> results struct |
| 5.6 | Create `df_run.m` | New: public API entry point | Dispatches to stage runners |
| 5.7 | Slim down the four MAIN scripts to thin wrappers | Each MAIN: build opts, call `df_run`, done | |
| 5.8 | Create `df.stages.compute_cost_statistics()` | Extract from `III_MAIN` lines 148-214 | Monte Carlo cost draws from identified set |
| 5.9 | Final validation: run all stages at fixture size, compare all outputs | `verify_refactor.m` | |
| 5.10 | Final validation: run all stages at full size, compare figures/tables to paper | Visual + diff | **Phase 5 gate: all paper outputs match; thin wrappers work** |

### Cleanup Session (1 session)

| # | Task | Notes |
|---|------|-------|
| C.1 | Move deprecated files to `matlab/src/_legacy/` | `ComputeBCCE_eps.m`, `_ApplicationL.m`, `_pass.m`, `epsilon_switch.m`, `epsilon_switch_distrib.m`, `find_polytope_switch.m`, `Identification_Pricing_Game_ApplicationL.m`, AMPL files |
| C.2 | Update `README.md` with new architecture and runtime expectations | |
| C.3 | Update `HANDOVER.md` with completed refactor status | |
| C.4 | Final timed production run at full scale | Record in `matlab/test/fixtures/timing_postrefactor.md` |
| C.5 | Git commit the full refactor | |

---

## 10. Session Summary

| Block | Sessions | Est. days | Deliverable |
|-------|:--------:|:---------:|-------------|
| Prep | 1 | 1 | Fixture oracle |
| Phase 1 (globals + setup) | 3 | 3-4 | Zero globals, `cfg` struct, `+df/` skeleton |
| Phase 2 (solver kernel) | 2-3 | 3-5 | Single `solve_bcce`, three old files deprecated |
| Phase 3a (coneprog) | 1 | 1-2 | Native SOCP backend, validated against CVX |
| Phase 3b (kill AMPL) | 1 | 1 | `linprog` replaces AMPL, AMPL files removed |
| Phase 4 (learning) | 2-3 | 3-5 | Fast learning kernel, `parfor` bootstrap |
| Phase 5 (compute/plot) | 2 | 2-3 | `df_run` API, thin wrappers, shared reporting |
| Cleanup | 1 | 1 | Legacy files archived, docs updated, final timing |
| **Total** | **13-17** | **15-22** | |
| Validation/benchmarking | 2-3 | 3-5 | Full-scale runs, backend cross-validation |
