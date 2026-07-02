%% II_RUN_hp_sensitivity  How sensitive is the high-prob region to the constant?
%  The switch_eps 12/13 high-probability constants are illustrative (the exact tail
%  constant is TBD with NL). This brackets where the real constant lands: take the
%  high-prob radius at N=4M and scale it by a multiplier, then report the region
%  area for RM (Hedge, sw12) and PRM (EXP3, sw13). A flat curve means the headline is
%  robust to the constant; a steep curve means the constant must be pinned down before
%  quoting magnitudes. Learns one RM and one PRM trajectory and re-solves via
%  eps_override = scale * compute_epsilon(...), fixed 60x60 grid, L1 consistency.
%  Incremental save per scale. NO figures.
clear; clc;
paths=df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; aR=0.045; aC=0.005; N=4000000; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
scales=[0.5 0.75 1.0 1.5 2.0 3.0];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',aR,'alpha_C',aC,'backend','fast', ...
    'adaptive',false,'require_corrected',true,'consistency_slack_kind','L1', ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

% reference high-prob radii (per-type, 1 x s) at N=4M
eps_rm=df.solvers.compute_epsilon(cfg,N,aR,12);   % HP Hedge
eps_pr=df.solvers.compute_epsilon(cfg,N,aR,13);   % HP EXP3

% learn one RM and one PRM trajectory; tiny 2x2 grid because only the trajectory
% is kept from these calls (the region solve is redone per scale below)
tiny=struct('fixed_gridparamM',[1;0.9;1.1],'fixed_gridparamV',[1;0.9;1.1]);
o=base; o.learning_style='rm'; o.switch_eps=10; o.fixed_gridparamM=tiny.fixed_gridparamM; o.fixed_gridparamV=tiny.fixed_gridparamV;
rng(12345); res=df.stages.run_stage_ii(cfg,o); traj_rm=res.distY_time_all;
o=base; o.learning_style='prm'; o.switch_eps=11; o.fixed_gridparamM=tiny.fixed_gridparamM; o.fixed_gridparamV=tiny.fixed_gridparamV;
rng(12345); res=df.stages.run_stage_ii(cfg,o); traj_pr=res.distY_time_all;

ns=numel(scales);
rm_area=nan(ns,1); rm_share=nan(ns,1); pr_area=nan(ns,1); pr_share=nan(ns,1);
out=fullfile(paths.artifacts,'hp_sensitivity.mat');
for k=1:ns
    sc=scales(k);
    % switch_eps must still be a corrected value (12/13) to pass the SIM-5
    % require_corrected guard; the radius itself comes from eps_override.
    o=base; o.learning_style='rm'; o.switch_eps=12; o.eps_override=sc*eps_rm; o.precomputed_distY=traj_rm;
    rng(12345); r=df.stages.run_stage_ii(cfg,o); [rm_area(k),rm_share(k)]=region_area(r,1,IDTOL);
    o=base; o.learning_style='prm'; o.switch_eps=13; o.eps_override=sc*eps_pr; o.precomputed_distY=traj_pr;
    rng(12345); r=df.stages.run_stage_ii(cfg,o); [pr_area(k),pr_share(k)]=region_area(r,1,IDTOL);
    fprintf('scale=%.2f  RM area=%.4f share=%.4f | PRM area=%.4f share=%.4f\n', ...
        sc, rm_area(k), rm_share(k), pr_area(k), pr_share(k));
    save(out,'scales','rm_area','rm_share','pr_area','pr_share','IDTOL','-v7.3');   % incremental
end
summary=table(scales(:), rm_area, rm_share, pr_area, pr_share, ...
    'VariableNames',{'scale','rm_area','rm_share','pr_area','pr_share'});
fprintf('\n=== high-prob constant sensitivity (N=4M, 60x60, L1) ===\n'); disp(summary);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'hp_sensitivity.csv'));
fprintf('Artifact: %s\n========== hp-sensitivity complete ==========\n', out);

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
