%% II_RUN_mc_coverage  (Niccolo Proposal 9: report empirical coverage)
%  Monte-Carlo coverage of the formal L1 region. For each of n_seeds independent DGP
%  draws, learn the RM play and test ONLY the true parameter (mu=3, sigma^2=1): a
%  1-point grid, so one SOCP per draw. Coverage = fraction of draws in which the true
%  parameter is retained. Nominal 0.95; over-coverage quantifies the conservativeness
%  that the tightening proposals aim to remove. NO figures.
clear; clc;
paths = df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL=1e-8; N=500000; n_seeds=200;
fixed1=[1];   % 1-point grid: multiplier 1 -> the true (mu,sigma^2)
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'switch_eps',10,'backend','fast','adaptive',false,'require_corrected',true, ...
    'learning_style','rm','consistency_slack_kind','L1', ...
    'fixed_gridparamV',fixed1,'fixed_gridparamM',fixed1);

covered=false(n_seeds,1); VV_true=nan(n_seeds,1);
out_path=fullfile(paths.artifacts,'mc_coverage.mat');
for k=1:n_seeds
    rng(k);                                   % independent DGP draw
    res=df.stages.run_stage_ii(cfg,base);
    VV_true(k)=res.VV_all(1,1); covered(k)=VV_true(k)<=IDTOL;
    if mod(k,20)==0
        fprintf('  seed %d/%d: running coverage = %.3f\n', k, n_seeds, mean(covered(1:k)));
        save(out_path,'covered','VV_true','N','n_seeds','IDTOL','-v7.3');   % incremental
    end
end
cov=mean(covered); se=sqrt(cov*(1-cov)/n_seeds);
fprintf('\n=== Prop 9: MC coverage of the L1 region (N=%dk, %d seeds) ===\n', N/1000, n_seeds);
fprintf('Empirical coverage = %.3f (SE %.3f); nominal 0.95\n', cov, se);
save(out_path,'covered','VV_true','N','n_seeds','cov','se','IDTOL','-v7.3');
fprintf('Artifact: %s\n========== MC coverage complete ==========\n', out_path);
