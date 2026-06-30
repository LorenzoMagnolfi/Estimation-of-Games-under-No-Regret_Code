%% II_RUN_looseness_decomp  (Niccolo Proposal 5: decompose sources of looseness)
%  Four regions on one cached RM trajectory (canonical lab, N=4M, 60x60 grid):
%    exact     : exact obedience (eps=0)        + exact consistency  (BCCE benchmark)
%    obed_only : relaxed obedience (eps at aR)  + exact consistency  (isolates regret)
%    cons_only : exact obedience (eps=0)        + L1 consistency     (isolates composition)
%    full      : relaxed obedience (eps at aR)  + L1 consistency     (formal region)
%  Shows immediately which component drives the width. Reports area per region. NO figures.
clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL=1e-8; aR=0.045; aC=0.005; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',4000000,'alpha_set',0.05,'switch_eps',10,'backend','fast', ...
    'adaptive',false,'require_corrected',true,'learning_style','rm', ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

% full learns the trajectory; the others reuse it
o=base; o.alpha_R=aR; o.alpha_C=aC; o.consistency_slack_kind='L1';
rng(12345); res_full=df.stages.run_stage_ii(cfg,o); traj=res_full.distY_time_all;

o=base; o.alpha_R=aR; o.consistency_slack_kind='none'; o.precomputed_distY=traj;
rng(12345); res_obed=df.stages.run_stage_ii(cfg,o);                          % relaxed obed + exact cons

o=base; o.alpha_R=aR; o.alpha_C=aC; o.consistency_slack_kind='L1'; o.eps_override=0; o.precomputed_distY=traj;
rng(12345); res_cons=df.stages.run_stage_ii(cfg,o);                          % exact obed + L1 cons

o=base; o.consistency_slack_kind='none'; o.eps_override=0; o.precomputed_distY=traj;
rng(12345); res_exact=df.stages.run_stage_ii(cfg,o);                          % exact obed + exact cons

names={'exact','obed_only','cons_only','full'};
R={res_exact,res_obed,res_cons,res_full};
area=zeros(4,1); share=zeros(4,1);
for k=1:4, [area(k),share(k)]=region_area(R{k},1,IDTOL); end
summary=table(names(:), area, share, 'VariableNames',{'region','area','share'});
fprintf('\n=== Prop 5: looseness decomposition (N=4M) ===\n'); disp(summary);

if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'looseness_decomp.csv'));
save(fullfile(paths.artifacts,'looseness_decomp.mat'),'res_exact','res_obed','res_cons','res_full','summary','IDTOL','-v7.3');
fprintf('Artifact: %s\n========== looseness decomposition complete ==========\n', fullfile(paths.artifacts,'looseness_decomp.mat'));

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
