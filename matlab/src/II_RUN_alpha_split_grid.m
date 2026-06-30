%% II_RUN_alpha_split_grid  (Niccolo Proposal 2: optimize the alpha_R/alpha_C split)
%  Sweep the obedience budget alpha_R over {0.030,0.035,0.040,0.045,0.0475}
%  (alpha_C = 0.05 - alpha_R), all at L1 consistency, full-feedback RM, canonical lab,
%  N = 4M. Reports area + mu/sigma projections per split, and the UNION over splits
%  (formally valid: each region covers at 1-alpha, so the union covers). Learns ONE
%  RM trajectory and re-solves under each split. NO figures.
clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL=1e-8; alpha_set=0.05; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
aR_grid=[0.030 0.035 0.040 0.045 0.0475];
base=struct('maxiters_values',4000000,'alpha_set',alpha_set,'switch_eps',10,'backend','fast', ...
    'adaptive',false,'require_corrected',true,'learning_style','rm','consistency_slack_kind','L1', ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

o=base; o.alpha_R=aR_grid(1); o.alpha_C=alpha_set-aR_grid(1);
rng(12345); res1=df.stages.run_stage_ii(cfg,o); traj=res1.distY_time_all;
nS=numel(aR_grid); res_by_split=cell(nS,1); res_by_split{1}=res1;
for k=2:nS
    o=base; o.alpha_R=aR_grid(k); o.alpha_C=alpha_set-aR_grid(k); o.precomputed_distY=traj;
    rng(12345); res_by_split{k}=df.stages.run_stage_ii(cfg,o);
end

nV=res1.NGridV+1; nM=res1.NGridM+1;
dp=squeeze(res1.distpars_all(1,:,:));
mu_g=reshape(dp(:,1),nV,nM); mu_g=mu_g(2:end,2:end);
sg_g=reshape(dp(:,2),nV,nM); sg_g=sg_g(2:end,2:end);
dmu=median(diff(unique(mu_g(:))),'omitnan'); dsg=median(diff(unique(sg_g(:))),'omitnan');
area=zeros(nS,1); mu_lo=nan(nS,1); mu_hi=nan(nS,1); sg_lo=nan(nS,1); sg_hi=nan(nS,1);
union_mask=false(size(mu_g));
for k=1:nS
    VV=reshape(res_by_split{k}.VV_all(1,:),nV,nM); VV=VV(2:end,2:end); id=VV<=IDTOL;
    area(k)=nnz(id)*abs(dmu)*abs(dsg);
    if any(id(:)), mu_lo(k)=min(mu_g(id)); mu_hi(k)=max(mu_g(id)); sg_lo(k)=min(sg_g(id)); sg_hi(k)=max(sg_g(id)); end
    union_mask=union_mask|id;
end
union_area=nnz(union_mask)*abs(dmu)*abs(dsg);
summary=table(aR_grid(:), alpha_set-aR_grid(:), area, mu_lo, mu_hi, sg_lo, sg_hi, ...
    'VariableNames',{'alpha_R','alpha_C','area','mu_lo','mu_hi','sg_lo','sg_hi'});
fprintf('\n=== Prop 2: alpha_R/alpha_C split sweep (N=4M, L1) ===\n'); disp(summary);
fprintf('UNION over the 5 splits: area = %.4f\n', union_area);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'alpha_split_grid.csv'));
save(fullfile(paths.artifacts,'alpha_split_grid.mat'),'res_by_split','aR_grid','summary','union_area','union_mask','IDTOL','-v7.3');
fprintf('Artifact: %s\n========== alpha-split grid complete ==========\n', fullfile(paths.artifacts,'alpha_split_grid.mat'));
