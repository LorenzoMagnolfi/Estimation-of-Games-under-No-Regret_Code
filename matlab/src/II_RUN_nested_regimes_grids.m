%% II_RUN_nested_regimes_grids  Export the per-candidate grids behind Figure S1.
%  II_RUN_nested_regimes (job 3788887) saved only summary areas; the sketch's
%  Figure S1 needs the full (mu, sigma2, VV) grid for the two Markov arms on
%  the SAME full-feedback trajectory. The learners are deterministic mean
%  dynamics, so re-learning with the same inputs reproduces job 3788887's
%  trajectory and regions exactly. Writes a compact per-point CSV for local
%  figure rendering: one row per candidate, VV under the bandit-class radius
%  (switch 11) and under the full-feedback radius (switch 10).
clear; clc;
paths=df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=4000000; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','learning_style','rm','use_parfor',true, ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

o=base; o.switch_eps=10; o.learn_only=true;
rng(12345); res=df.stages.run_stage_ii(cfg,o); traj=res.distY_time_all;

o=base; o.switch_eps=11; o.precomputed_distY=traj;
rng(12345); r_bd=df.stages.run_stage_ii(cfg,o);
o=base; o.switch_eps=10; o.precomputed_distY=traj;
rng(12345); r_ff=df.stages.run_stage_ii(cfg,o);

dp=squeeze(r_ff.distpars_all(1,:,:));
mu_c=dp(:,1); sg_c=dp(:,2);
vv_bandit=r_bd.VV_all(1,:).'; vv_ff=r_ff.VV_all(1,:).';
T=table(mu_c,sg_c,vv_bandit,vv_ff, ...
    'VariableNames',{'mu','sigma2','vv_bandit_markov','vv_ff_markov'});
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(T, fullfile(paths.tables_ii,'nested_regimes_grids.csv'));
save(fullfile(paths.artifacts,'nested_regimes_grids.mat'), ...
    'mu_c','sg_c','vv_bandit','vv_ff','IDTOL','-v7.3');
n_bd=nnz(vv_bandit<=IDTOL); n_ff=nnz(vv_ff<=IDTOL);
fprintf('grids exported: bandit identified %d, full-feedback %d of %d\n', ...
    n_bd, n_ff, numel(vv_ff));
fprintf('========== nested-regimes grids complete ==========\n');
