%% II_RUN_v05_rate  Area-vs-N under the V05 calibrations, Hedge and EXP3-IX.
%  For each N in {0.5,1,2,4,8}M: learn one Hedge and one EXP3-IX trajectory (seed 12345),
%  then solve on a fixed 40x40 grid (areas comparable across N), L1 consistency:
%    Hedge   16 momentF | 14 tailF | 10 legacy per-cell Markov (no M_R; continuity only)
%    EXP3-IX 17 momentB | 15 tailB
%  Incremental save per N. Realized regrets saved per learner and N. No figures.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; NGRID=40; Ns=[500000 1000000 2000000 4000000 8000000];
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false, ...
    'require_corrected',true,'consistency_slack_kind','L1','use_parfor',true, ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);
arms={'hedge_momentF',16,'hedge'; 'hedge_tailF',14,'hedge'; 'hedge_legacyMarkov10',10,'hedge'; ...
      'exp3ix_momentB',17,'exp3ix'; 'exp3ix_tailB',15,'exp3ix'};
out=fullfile(paths.artifacts,'v05_rate.mat');

N_col=[]; arm_col={}; learner_col={}; sw_col=[]; area_col=[]; share_col=[]; hb_col=[]; maxreg_col=[];
for ni=1:numel(Ns)
    Nv=Ns(ni);
    rng(12345); t0=tic; [dY_h,~,fr_h]=df.sim.learn_hedge(cfg,1,Nv,Nv,1,1,1,1);
    fprintf('N=%dk hedge learned %.0fs, max realized regret %.3e\n', Nv/1000, toc(t0), max(fr_h(:)));
    rng(12345); t0=tic; [dY_x,~,fr_x]=df.sim.learn_exp3ix(cfg,1,Nv,Nv,1,1,1,1);
    fprintf('N=%dk exp3ix learned %.0fs, max realized regret %.3e\n', Nv/1000, toc(t0), max(fr_x(:)));
    traj=struct('hedge',{{dY_h}},'exp3ix',{{dY_x}}); maxreg=struct('hedge',max(fr_h(:)),'exp3ix',max(fr_x(:)));
    for a=1:size(arms,1)
        nm=arms{a,1}; sw=arms{a,2}; sty=arms{a,3};
        o=base; o.maxiters_values=Nv; o.learning_style=sty; o.switch_eps=sw; o.precomputed_distY=traj.(sty);
        rng(12345); r=df.stages.run_stage_ii(cfg,o);
        [ar,sh,hb]=region_area(r,1,IDTOL);
        fprintf('N=%6dk %-22s area=%.4f share=%.4f hb=%d\n', Nv/1000, nm, ar, sh, hb);
        N_col(end+1)=Nv; arm_col{end+1}=nm; learner_col{end+1}=sty; sw_col(end+1)=sw; %#ok<SAGROW>
        area_col(end+1)=ar; share_col(end+1)=sh; hb_col(end+1)=hb; maxreg_col(end+1)=maxreg.(sty); %#ok<SAGROW>
    end
    save(out,'N_col','arm_col','learner_col','sw_col','area_col','share_col','hb_col','maxreg_col','Ns','IDTOL','-v7.3');
end
summary=table(N_col(:), arm_col(:), learner_col(:), sw_col(:), area_col(:), share_col(:), logical(hb_col(:)), maxreg_col(:), ...
    'VariableNames',{'N','arm','learner','switch_eps','area','share','hit_boundary','max_realized_regret'});
fprintf('\n=== V05 area-vs-N (40x40, L1) ===\n'); disp(summary);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'v05_rate.csv'));
fprintf('Artifact: %s\n========== V05 rate complete ==========\n', out);

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
