%% II_RUN_mc_area_quantiles  Sampling distribution of the region area (memo §10.3).
%  Every headline area so far is a single-seed number. This runs 50 independent
%  full-feedback trajectories at the headline horizon N=4M and reports the
%  Monte Carlo distribution (median, quartiles, 5/95) of the identified-region
%  area under the Markov radius and, for reference at the illustrative
%  constant, the high-probability radius. 41x41 fixed grid, L1 consistency,
%  direct lane + parfor; checkpoint save per seed.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=4000000; NGRID=40; NSEEDS=50;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','learning_style','rm','use_parfor',true, ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

seed=(1:NSEEDS)'; area_mk=nan(NSEEDS,1); share_mk=nan(NSEEDS,1);
area_hp=nan(NSEEDS,1); share_hp=nan(NSEEDS,1);
out=fullfile(paths.artifacts,'mc_area_quantiles.mat');
for k=1:NSEEDS
    o=base; o.switch_eps=10; o.learn_only=true;
    rng(seed(k)); res=df.stages.run_stage_ii(cfg,o); traj=res.distY_time_all;
    o=base; o.switch_eps=10; o.precomputed_distY=traj;
    r=df.stages.run_stage_ii(cfg,o); [area_mk(k),share_mk(k)]=region_area(r,1,IDTOL);
    o=base; o.switch_eps=12; o.precomputed_distY=traj;
    r=df.stages.run_stage_ii(cfg,o); [area_hp(k),share_hp(k)]=region_area(r,1,IDTOL);
    fprintf('seed %2d/%d: markov area=%.4f  hp area=%.4f\n', k, NSEEDS, area_mk(k), area_hp(k));
    save(out,'seed','area_mk','share_mk','area_hp','share_hp','IDTOL','-v7.3');
end
q=@(v,p) quantile(v,p);
fprintf('\n=== MC area distribution over %d seeds (N=4M, L1) ===\n', NSEEDS);
fprintf('Markov: median=%.4f  IQR=[%.4f, %.4f]  5/95=[%.4f, %.4f]\n', ...
    q(area_mk,.5), q(area_mk,.25), q(area_mk,.75), q(area_mk,.05), q(area_mk,.95));
fprintf('HP(illustr.): median=%.5f  IQR=[%.5f, %.5f]  5/95=[%.5f, %.5f]\n', ...
    q(area_hp,.5), q(area_hp,.25), q(area_hp,.75), q(area_hp,.05), q(area_hp,.95));
summary=table(seed,area_mk,share_mk,area_hp,share_hp);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'mc_area_quantiles.csv'));
fprintf('Artifact: %s\n========== MC area quantiles complete ==========\n', out);

function [area,share]=region_area(res,ni,IDTOL)
    nV=res.NGridV+1; nM=res.NGridM+1;
    dp=squeeze(res.distpars_all(ni,:,:));
    mu_g=reshape(dp(:,1),nV,nM); mu_g=mu_g(2:end,2:end);
    sg_g=reshape(dp(:,2),nV,nM); sg_g=sg_g(2:end,2:end);
    VV=reshape(res.VV_all(ni,:),nV,nM); VV=VV(2:end,2:end); id=VV<=IDTOL;
    share=nnz(id)/numel(id);
    dmu=median(diff(unique(mu_g(:))),'omitnan'); dsg=median(diff(unique(sg_g(:))),'omitnan');
    area=nnz(id)*abs(dmu)*abs(dsg);
end
