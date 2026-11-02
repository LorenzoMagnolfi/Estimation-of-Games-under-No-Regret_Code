%% II_RUN_v05_coverage  Coverage of the true parameter under the V05 calibrations, 200 seeds.
%  For each seed: learn ONE Hedge and ONE EXP3-IX trajectory at N = 500k (independent DGP
%  draw; the same seed gives both learners the same cost sequence), then test the true
%  (mu, sigma^2) = (3, 1) on a one-point grid under
%    Hedge   16 momentF | 14 tailF      EXP3-IX 17 momentB | 15 tailB
%  with L1 consistency. Reports empirical coverage (nominal 0.95) and the realized
%  regrets. Incremental save every 20 seeds. No figures.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=500000; n_seeds=200; fixed1=1;
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true,'use_parfor',false, ...
    'consistency_slack_kind','L1','fixed_gridparamV',fixed1,'fixed_gridparamM',fixed1);
arms={'hedge_momentF',16,'hedge'; 'hedge_tailF',14,'hedge'; 'exp3ix_momentB',17,'exp3ix'; 'exp3ix_tailB',15,'exp3ix'};
n_arms=size(arms,1);
cov=false(n_seeds,n_arms); VV=nan(n_seeds,n_arms); maxreg=nan(n_seeds,2);
out=fullfile(paths.artifacts,'v05_coverage.mat');
for k=1:n_seeds
    rng(k); [dY_h,~,fr_h]=df.sim.learn_hedge(cfg,1,N,N,1,1,1,1);
    rng(k); [dY_x,~,fr_x]=df.sim.learn_exp3ix(cfg,1,N,N,1,1,1,1);
    traj=struct('hedge',{{dY_h}},'exp3ix',{{dY_x}}); maxreg(k,:)=[max(fr_h(:)), max(fr_x(:))];
    for a=1:n_arms
        o=base; o.switch_eps=arms{a,2}; o.learning_style=arms{a,3}; o.precomputed_distY=traj.(arms{a,3});
        r=df.stages.run_stage_ii(cfg,o);
        VV(k,a)=r.VV_all(1,1); cov(k,a)=VV(k,a)<=IDTOL;
    end
    if mod(k,20)==0
        fprintf('  seed %d/%d: coverage %s\n', k, n_seeds, mat2str(mean(cov(1:k,:),1),3));
        save(out,'cov','VV','maxreg','N','n_seeds','IDTOL','arms','-v7.3');
    end
end
c=mean(cov,1); se=sqrt(c.*(1-c)/n_seeds);
fprintf('\n=== V05 coverage (N=%dk, %d seeds, nominal 0.95) ===\n', N/1000, n_seeds);
for a=1:n_arms, fprintf('%-16s coverage = %.3f (SE %.3f)\n', arms{a,1}, c(a), se(a)); end
fprintf('realized regret (max over types): hedge median %.3e, exp3ix median %.3e\n', median(maxreg(:,1)), median(maxreg(:,2)));
summary=table(arms(:,1), c(:), se(:), 'VariableNames', {'arm','coverage','se'});
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'v05_coverage.csv'));
save(out,'cov','VV','maxreg','N','n_seeds','c','se','IDTOL','arms','-v7.3');
fprintf('Artifact: %s\n========== V05 coverage complete ==========\n', out);
