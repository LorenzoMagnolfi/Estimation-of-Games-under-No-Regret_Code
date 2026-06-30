%% II_RUN_coverage_hp  Coverage of the Markov vs high-probability region.
%  Prop 9 found the Markov L1 region over-covers (empirical 1.000 vs nominal 0.95).
%  Prop 3 showed the high-probability radius shrinks that region ~120x. The shrinkage
%  is only legitimate if coverage stays >= 0.95. For each of n_seeds independent DGP
%  draws, learn ONE RM trajectory and test the true parameter (mu=3, sigma^2=1) under
%  BOTH radii (same trajectory): switch_eps=10 (Markov) and switch_eps=12 (high-prob,
%  Hedge), L1 consistency, N=500k. Reports empirical coverage of each. If high-prob
%  coverage drops below nominal, the illustrative HP constant is too aggressive. NO figures.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=500000; n_seeds=200; fixed1=1;     % 1-point grid: the true (mu,sigma^2)
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true,'learning_style','rm', ...
    'consistency_slack_kind','L1','fixed_gridparamV',fixed1,'fixed_gridparamM',fixed1);

cov_mk=false(n_seeds,1); cov_hp=false(n_seeds,1); VV_mk=nan(n_seeds,1); VV_hp=nan(n_seeds,1);
out=fullfile(paths.artifacts,'coverage_hp.mat');
for k=1:n_seeds
    rng(k);                                       % independent DGP draw
    o=base; o.switch_eps=10; res_mk=df.stages.run_stage_ii(cfg,o); traj=res_mk.distY_time_all;
    o=base; o.switch_eps=12; o.precomputed_distY=traj; res_hp=df.stages.run_stage_ii(cfg,o);
    VV_mk(k)=res_mk.VV_all(1,1); cov_mk(k)=VV_mk(k)<=IDTOL;
    VV_hp(k)=res_hp.VV_all(1,1); cov_hp(k)=VV_hp(k)<=IDTOL;
    if mod(k,20)==0
        fprintf('  seed %d/%d: cov_Markov=%.3f  cov_HighProb=%.3f\n', ...
            k, n_seeds, mean(cov_mk(1:k)), mean(cov_hp(1:k)));
        save(out,'cov_mk','cov_hp','VV_mk','VV_hp','N','n_seeds','IDTOL','-v7.3');
    end
end
c_mk=mean(cov_mk); c_hp=mean(cov_hp);
se=@(c) sqrt(c*(1-c)/n_seeds);
fprintf('\n=== Prop 3/9: coverage Markov vs high-prob (N=%dk, %d seeds, nominal 0.95) ===\n', N/1000, n_seeds);
fprintf('Markov   coverage = %.3f (SE %.3f)\n', c_mk, se(c_mk));
fprintf('HighProb coverage = %.3f (SE %.3f)\n', c_hp, se(c_hp));
save(out,'cov_mk','cov_hp','VV_mk','VV_hp','N','n_seeds','c_mk','c_hp','IDTOL','-v7.3');
fprintf('Artifact: %s\n========== coverage Markov-vs-HP complete ==========\n', out);
