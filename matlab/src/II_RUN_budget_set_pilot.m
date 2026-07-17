%% II_RUN_budget_set_pilot  Joint obedience budget vs componentwise rectangle.
%  NL's architecture memo (17 Jul) makes a joint regret uncertainty set the
%  preferred Theorem 3 route. This pilot quantifies the geometry on one cached
%  full-feedback trajectory at N=4M: the standard rectangle (per-type caps,
%  switch 10, L1 consistency) against the budget set that replaces the caps
%  with ONE per-candidate budget B(lambda) = scale * sum of the rectangle's own
%  weighted caps. scale=1 contains the rectangle by construction (same total,
%  trade-offs allowed); smaller scales trace where the joint set crosses the
%  rectangle's area -- the curve NL needs to size the union-bound saving.
%  Conventions note: the per-type caps carry the code's Markov convention as-is;
%  the M_R placement question is orthogonal and flagged in the crosswalk memo.
%  CVX lane both arms (the budget max-term is not in the direct lane yet).
clear; clc;
paths=df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=4000000; NGRID=40;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'switch_eps',10,'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','learning_style','rm','solver_backend','cvx', ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

o=base; o.learn_only=true; rng(12345); res=df.stages.run_stage_ii(cfg,o); traj=res.distY_time_all;

out=fullfile(paths.artifacts,'budget_set_pilot.mat');
NGrid=(NGRID+1)^2;
scales=[1.0 0.4 0.2 0.1];
arm=cell(0,1); scl=[]; area=[]; share=[];

% rectangle arm (standard componentwise caps)
o=base; o.precomputed_distY=traj; rng(12345); r=df.stages.run_stage_ii(cfg,o);
[ar,sh]=region_area(r,1,IDTOL);
arm{end+1,1}='rectangle'; scl(end+1,1)=NaN; area(end+1,1)=ar; share(end+1,1)=sh;
fprintf('rectangle: area=%.4f share=%.4f\n', ar, sh);
save(out,'arm','scl','area','share','IDTOL','-v7.3');

% budget arms, descending scale with pruning (region monotone increasing in B)
alive=true(NGrid,1);
for k=1:numel(scales)
    o=base; o.precomputed_distY=traj; o.obed_budget_scale=scales(k);
    o.candidate_mask=alive;
    rng(12345); r=df.stages.run_stage_ii(cfg,o);
    vv=r.VV_all(1,:); vv(~alive.')=100;
    rr=r; rr.VV_all(1,:)=vv; [ar,sh]=region_area(rr,1,IDTOL);
    arm{end+1,1}=sprintf('budget'); scl(end+1,1)=scales(k); area(end+1,1)=ar; share(end+1,1)=sh; %#ok<SAGROW>
    fprintf('budget scale=%.2f: area=%.4f share=%.4f (solved %d/%d)\n', ...
        scales(k), ar, sh, nnz(o.candidate_mask), NGrid);
    alive=(vv.'<=IDTOL);
    save(out,'arm','scl','area','share','IDTOL','-v7.3');
end
% containment check: budget at scale 1 must contain the rectangle
a_rect=area(strcmp(arm,'rectangle')); a_b1=area(scl==1.0);
fprintf('containment (budget(1) >= rectangle): %.4f >= %.4f -> %s\n', ...
    a_b1, a_rect, string(a_b1 >= a_rect - 1e-9));
summary=table(arm,scl,area,share);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'budget_set_pilot.csv'));
fprintf('Artifact: %s\n========== budget-set pilot complete ==========\n', out);

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
