%% II_RUN_v05_frontier  One trajectory per certified learner; the V05 calibrations on the same data.
%  Hedge (family F, df.sim.learn_hedge) and EXP3-IX (family B, df.sim.learn_exp3ix) at
%  N = 4M in the canonical two-seller lab; 61x61 fixed (mu, sigma) grid; L1 consistency
%  (alpha_R = 0.045, alpha_C = 0.005). Arms (switch_eps):
%    Hedge   16 momentF  first-moment rectangle at the expected count, with M_R/alpha_R
%            14 tailF    direct tail at delta = alpha_R/M_R, latent-count ceiling
%            10 legacy per-cell Markov WITHOUT the M_R factor (July convention; continuity only)
%            12 legacy illustrative high-probability radius (July; continuity only)
%    EXP3-IX 17 momentB  first-moment rectangle (calendar bound, keeps its sqrt(N) term)
%            15 tailB    direct tail
%  Same trajectory within each learner, so nesting across calibrations is interpretable.
%  Realized regrets of both learners are saved. Incremental save per arm; CSV; no figures.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=4000000; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','use_parfor',true, ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);
out=fullfile(paths.artifacts,'v05_frontier.mat');
traj_file=fullfile(paths.artifacts,'v05_traj_4M.mat');

% Learn once per learner (Stage II convention: N = 1 random period, M = N periods of learning).
rng(12345); t0=tic; [dY_h,~,fr_hedge]=df.sim.learn_hedge(cfg,1,N,N,1,1,1,1); t_h=toc(t0);
rng(12345); t0=tic; [dY_x,~,fr_exp3ix]=df.sim.learn_exp3ix(cfg,1,N,N,1,1,1,1); t_x=toc(t0);
traj=struct('hedge',{{dY_h}},'exp3ix',{{dY_x}});
fprintf('learned hedge %.0fs (max realized regret %.3e), exp3ix %.0fs (max realized regret %.3e)\n', ...
    t_h, max(fr_hedge(:)), t_x, max(fr_exp3ix(:)));
save(traj_file,'dY_h','dY_x','fr_hedge','fr_exp3ix','N','-v7.3');

arms={'hedge_momentF',16,'hedge'; 'hedge_tailF',14,'hedge'; 'hedge_legacyMarkov10',10,'hedge'; ...
      'hedge_legacyHP12',12,'hedge'; 'exp3ix_momentB',17,'exp3ix'; 'exp3ix_tailB',15,'exp3ix'};
n=size(arms,1); name=cell(n,1); sw=zeros(n,1); learner=cell(n,1);
area=zeros(n,1); share=zeros(n,1); hb=zeros(n,1); VV=cell(n,1); grids=cell(n,1); t_solve=zeros(n,1);
for k=1:n
    o=base; o.switch_eps=arms{k,2}; o.learning_style=arms{k,3}; o.precomputed_distY=traj.(arms{k,3});
    rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); t_solve(k)=toc(t0);
    [area(k),share(k),hb(k)]=region_area(r,1,IDTOL);
    name{k}=arms{k,1}; sw(k)=arms{k,2}; learner{k}=arms{k,3};
    VV{k}=r.VV_all(1,:); grids{k}=squeeze(r.distpars_all(1,:,:));
    fprintf('%-22s (sw%2d): area=%.4f share=%.4f boundary=%d  [%.0fs]\n', name{k}, sw(k), area(k), share(k), hb(k), t_solve(k));
    save(out,'name','sw','learner','area','share','hb','VV','grids','IDTOL','N','fr_hedge','fr_exp3ix','t_solve','-v7.3');
end
summary=table(name,sw,learner,area,share,logical(hb),t_solve,'VariableNames',{'arm','switch_eps','learner','area','share','hit_boundary','solve_seconds'});
fprintf('\n=== V05 frontier at N=4M (61x61, L1) ===\n'); disp(summary);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'v05_frontier.csv'));
fprintf('Artifact: %s\n========== V05 frontier complete ==========\n', out);

function [area,share,hb]=region_area(res,ni,IDTOL)
    nV=res.NGridV+1; nM=res.NGridM+1;
    dp=squeeze(res.distpars_all(ni,:,:));
    mu_g=reshape(dp(:,1),nV,nM); mu_g=mu_g(2:end,2:end);
    sg_g=reshape(dp(:,2),nV,nM); sg_g=sg_g(2:end,2:end);
    VV=reshape(res.VV_all(ni,:),nV,nM); VV=VV(2:end,2:end); id=VV<=IDTOL;
    share=nnz(id)/numel(id);
    dmu=median(diff(unique(mu_g(:))),'omitnan'); dsg=median(diff(unique(sg_g(:))),'omitnan');
    area=nnz(id)*abs(dmu)*abs(dsg);
    hb=any(id(1,:))||any(id(end,:))||any(id(:,1))||any(id(:,end));
end
