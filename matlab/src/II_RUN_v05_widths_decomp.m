%% II_RUN_v05_widths_decomp  (a) projection widths on a fine local grid; (b) width decomposition under the tail.
%  (a) Hedge (tailF, momentF) and EXP3-IX (tailB) at N in {1, 2, 4}M on a 41x41 LOCAL grid around the
%      truth (mu in [2.79, 3.21], sigma^2 in [0.80, 1.25]); reports area, share, and the projection widths
%      in mu and sigma^2 of the feasible lattice cells, plus truth feasibility. This resolves the regions
%      that sit below the 40x40 and 61x61 cells of the frontier and rate runs.
%  (b) Decomposition of the width at N = 4M on the 61x61 grid, per learner, on one cached trajectory:
%        exact      exact obedience + exact consistency      (BCCE benchmark)
%        obed_only  tail obedience + exact consistency       (diagnostic: tail at the candidate's shares)
%        cons_only  exact obedience + L1 consistency         (isolates composition)
%        full       tail obedience + L1 consistency          (the formal region)
%      July's version used the legacy radius; this one uses the certified tails (codes 14, 15).
%  Incremental saves; CSVs; no figures.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8;
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false, ...
    'require_corrected',true,'consistency_slack_kind','L1','use_parfor',true);

%% (a) local widths
NG=40; lgM=[1;linspace(0.93,1.07,NG)']; lgV=[1;linspace(0.80,1.25,NG)'];
Ns=[1000000 2000000 4000000];
out_a=fullfile(paths.artifacts,'v05_widths.mat');
W=table();
traj4=struct();
for ni=1:numel(Ns)
    Nv=Ns(ni);
    rng(12345); t0=tic; [dY_h,~,fr_h]=df.sim.learn_hedge(cfg,1,Nv,Nv,1,1,1,1); fprintf('N=%dk hedge learned %.0fs\n',Nv/1000,toc(t0));
    rng(12345); t0=tic; [dY_x,~,fr_x]=df.sim.learn_exp3ix(cfg,1,Nv,Nv,1,1,1,1); fprintf('N=%dk exp3ix learned %.0fs\n',Nv/1000,toc(t0));
    if Nv==4000000, traj4.dY_h=dY_h; traj4.dY_x=dY_x; end
    arms={'hedge',14,{dY_h},max(fr_h(:)); 'hedge',16,{dY_h},max(fr_h(:)); 'exp3ix',15,{dY_x},max(fr_x(:))};
    for a=1:size(arms,1)
        o=base; o.maxiters_values=Nv; o.learning_style=arms{a,1}; o.switch_eps=arms{a,2}; o.precomputed_distY=arms{a,3};
        o.fixed_gridparamV=lgV; o.fixed_gridparamM=lgM;
        rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); ts=toc(t0);
        [area,share,hb,wmu,wsg,ncell,truth_in]=region_widths(r,1,IDTOL);
        W=[W; table(Nv,string(arms{a,1}),arms{a,2},area,share,hb,wmu,wsg,ncell,truth_in,arms{a,4},ts, ...
            'VariableNames',{'N','learner','switch_eps','area','share','hit_boundary','width_mu','width_sigma2','n_cells','truth_feasible','max_realized_regret','solve_seconds'})]; %#ok<AGROW>
        fprintf('N=%dk %-7s sw%d: area=%.5f share=%.4f cells=%d width_mu=%.3f width_sigma2=%.3f hb=%d truth=%d [%.0fs]\n', ...
            Nv/1000, arms{a,1}, arms{a,2}, area, share, ncell, wmu, wsg, hb, truth_in, ts);
        save(out_a,'W','lgM','lgV','IDTOL','-v7.3');
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(W, fullfile(paths.tables_ii,'v05_widths.csv'));
fprintf('\n=== (a) local widths ===\n'); disp(W);

%% (b) decomposition at 4M, 61x61
NGRID=60; fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
out_b=fullfile(paths.artifacts,'v05_decomp.mat');
D=table();
learners={'hedge',14,{traj4.dY_h}; 'exp3ix',15,{traj4.dY_x}};
for L=1:size(learners,1)
    lrn=learners{L,1}; sw=learners{L,2}; tr=learners{L,3};
    variants={'exact',10,'none',0,false; 'obed_only',sw,'none',[],true; 'cons_only',10,'L1',0,false; 'full',sw,'L1',[],false};
    for v=1:size(variants,1)
        o=base; o.maxiters_values=4000000; o.learning_style=lrn; o.switch_eps=variants{v,2}; o.precomputed_distY=tr;
        o.consistency_slack_kind=variants{v,3}; o.eps_override=variants{v,4}; o.tail_allow_exact=variants{v,5};
        if strcmp(o.consistency_slack_kind,'none'), o.alpha_R=0.05; o.alpha_C=0; end
        o.fixed_gridparamV=fgV; o.fixed_gridparamM=fgM;
        rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); ts=toc(t0);
        [area,share,hb,wmu,wsg,ncell,truth_in]=region_widths(r,1,IDTOL);
        D=[D; table(string(lrn),string(variants{v,1}),area,share,hb,wmu,wsg,ncell,truth_in,ts, ...
            'VariableNames',{'learner','region','area','share','hit_boundary','width_mu','width_sigma2','n_cells','truth_feasible','solve_seconds'})]; %#ok<AGROW>
        fprintf('%-7s %-10s: area=%.4f share=%.4f cells=%d hb=%d truth=%d [%.0fs]\n', lrn, variants{v,1}, area, share, ncell, hb, truth_in, ts);
        save(out_b,'D','IDTOL','-v7.3');
    end
end
writetable(D, fullfile(paths.tables_ii,'v05_decomp.csv'));
fprintf('\n=== (b) width decomposition at N=4M (61x61) ===\n'); disp(D);
fprintf('Artifacts: %s, %s\n========== V05 widths + decomposition complete ==========\n', out_a, out_b);

function [area,share,hb,wmu,wsg,ncell,truth_in]=region_widths(res,ni,IDTOL)
    nV=res.NGridV+1; nM=res.NGridM+1;
    dp=squeeze(res.distpars_all(ni,:,:));
    mu_g=reshape(dp(:,1),nV,nM); mu_g=mu_g(2:end,2:end);
    sg_g=reshape(dp(:,2),nV,nM); sg_g=sg_g(2:end,2:end);
    VVfull=reshape(res.VV_all(ni,:),nV,nM); truth_in=VVfull(1,1)<=IDTOL;
    VV=VVfull(2:end,2:end); id=VV<=IDTOL;
    ncell=nnz(id); share=ncell/numel(id);
    dmu=median(diff(unique(mu_g(:))),'omitnan'); dsg=median(diff(unique(sg_g(:))),'omitnan');
    area=ncell*abs(dmu)*abs(dsg);
    hb=any(id(1,:))||any(id(end,:))||any(id(:,1))||any(id(:,end));
    if ncell>0
        wmu=max(mu_g(id))-min(mu_g(id))+abs(dmu); wsg=max(sg_g(id))-min(sg_g(id))+abs(dsg);   % cell-inclusive widths
    else
        wmu=0; wsg=0;
    end
end
