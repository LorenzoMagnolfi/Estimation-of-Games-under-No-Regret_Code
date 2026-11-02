%% II_RUN_v05_sweeps  Action-grid and cost-support sweeps under the certified calibrations.
%  (c) Action grid |A| in {5, 10, 20} (prices linspace(4, 8, |A|)), s = 5: Hedge tailF on the 41x41 LOCAL grid
%      at N = 1M and 4M (areas resolvable below the global cells); EXP3-IX tailB on the global grid at 4M
%      (60x60 / 40x40 / 20x20 as in July's sweep).            Referee: R1.1, R1.4.a.
%  (d) Cost support s in {5, 10, 20} (|A| = 5): Hedge tailF on the local grid at 1M and 4M; EXP3-IX tailB on
%      a 30x30 global grid at 4M (July's costsupport grid).    Referee: R1.4.b.
%  Trajectories are learned per (|A|, s) configuration with seed 12345. Incremental saves; CSVs; no figures.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); mu=3*ones(2,1); sigma2=eye(2);
IDTOL=1e-8;
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false, ...
    'require_corrected',true,'consistency_slack_kind','L1','use_parfor',true);
NG=40; lgM=[1;linspace(0.93,1.07,NG)']; lgV=[1;linspace(0.80,1.25,NG)'];
out=fullfile(paths.artifacts,'v05_sweeps.mat');
S=table();

function_rows = {};  %#ok<NASGU>
configs = {};
for A=[5 10 20], configs{end+1}=struct('tag','A','A',A,'s',5); end %#ok<SAGROW>
for s=[10 20],   configs{end+1}=struct('tag','s','A',5,'s',s); end %#ok<SAGROW>   % s = 5 is the |A| = 5 row above
for ci=1:numel(configs)
    c=configs{ci}; actions_vec=linspace(4,8,c.A)';
    cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,c.s);
    switch c.A, case 5, G=60; case 10, G=40; otherwise, G=20; end
    if c.s>5, G=30; end
    fgM=[1;linspace(0.18,1.5,G)']; fgV=[1;linspace(0.15,3.5,G)'];
    fprintf('\n===== |A|=%d, s=%d =====\n', c.A, c.s);
    for Nv=[1000000 4000000]
        rng(12345); t0=tic; [dY_h,~,fr_h]=df.sim.learn_hedge(cfg,1,Nv,Nv,1,1,1,1); fprintf('N=%dk hedge learned %.0fs\n',Nv/1000,toc(t0));
        arms={'hedge',14,{dY_h},max(fr_h(:)),'local'};
        if Nv==4000000
            rng(12345); t0=tic; [dY_x,~,fr_x]=df.sim.learn_exp3ix(cfg,1,Nv,Nv,1,1,1,1); fprintf('N=%dk exp3ix learned %.0fs\n',Nv/1000,toc(t0));
            arms(end+1,:)={'exp3ix',15,{dY_x},max(fr_x(:)),'global'}; %#ok<SAGROW>
        end
        for a=1:size(arms,1)
            o=base; o.maxiters_values=Nv; o.learning_style=arms{a,1}; o.switch_eps=arms{a,2}; o.precomputed_distY=arms{a,3};
            if strcmp(arms{a,5},'local'), o.fixed_gridparamV=lgV; o.fixed_gridparamM=lgM; else, o.fixed_gridparamV=fgV; o.fixed_gridparamM=fgM; end
            rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); ts=toc(t0);
            [area,share,hb,wmu,wsg,ncell,truth_in]=region_widths(r,1,IDTOL);
            S=[S; table(string(c.tag),c.A,c.s,Nv,string(arms{a,1}),arms{a,2},string(arms{a,5}),area,share,hb,wmu,wsg,ncell,truth_in,arms{a,4},ts, ...
                'VariableNames',{'sweep','A','s','N','learner','switch_eps','grid','area','share','hit_boundary','width_mu','width_sigma2','n_cells','truth_feasible','max_realized_regret','solve_seconds'})]; %#ok<AGROW>
            fprintf('|A|=%2d s=%2d N=%dk %-7s sw%d %-6s: area=%.5f share=%.4f cells=%d width_mu=%.3f width_sigma2=%.3f hb=%d truth=%d [%.0fs]\n', ...
                c.A, c.s, Nv/1000, arms{a,1}, arms{a,2}, arms{a,5}, area, share, ncell, wmu, wsg, hb, truth_in, ts);
            save(out,'S','lgM','lgV','IDTOL','-v7.3');
        end
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(S, fullfile(paths.tables_ii,'v05_sweeps.csv'));
fprintf('\n=== V05 sweeps ===\n'); disp(S);
fprintf('Artifact: %s\n========== V05 sweeps complete ==========\n', out);

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
        wmu=max(mu_g(id))-min(mu_g(id))+abs(dmu); wsg=max(sg_g(id))-min(sg_g(id))+abs(dsg);
    else
        wmu=0; wsg=0;
    end
end
