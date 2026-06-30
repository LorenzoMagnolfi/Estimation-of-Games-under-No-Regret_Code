%% II_RUN_hp_vs_N  Area-vs-N under Markov vs high-probability radius, RM and PRM.
%  Prop 3 compared the radii at a single N (4M). This traces the whole curve so we
%  can see WHERE the bandit (PRM) region becomes informative under the high-prob
%  envelope. For each N in {0.5,1,2,4,8}M: learn one RM and one PRM trajectory, then
%  solve four arms on a fixed 40x40 grid (so areas are comparable across N), all with
%  L1 consistency:
%    RM Markov (sw10) | RM HighProb (sw12) | PRM Markov (sw11) | PRM HighProb (sw13).
%  Incremental save per N. NO figures.
%  NOTE: the 12/13 high-prob constants are ILLUSTRATIVE (TBD with NL); this shows the
%  SHAPE of the informativeness gain, not a final radius.
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; NGRID=40; Ns=[500000 1000000 2000000 4000000 8000000];
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false, ...
    'require_corrected',true,'consistency_slack_kind','L1','fixed_gridparamV',fgV,'fixed_gridparamM',fgM);
arms={'RM_Markov',10,'rm'; 'RM_HighProb',12,'rm'; 'PRM_Markov',11,'prm'; 'PRM_HighProb',13,'prm'};
out=fullfile(paths.artifacts,'hp_vs_N.mat');

N_col=[]; arm_col={}; area_col=[]; share_col=[]; hb_col=[];
for ni=1:numel(Ns)
    Nv=Ns(ni);
    % learn one RM and one PRM trajectory at this N
    o=base; o.maxiters_values=Nv; o.learning_style='rm'; o.switch_eps=10;
    rng(12345); res=df.stages.run_stage_ii(cfg,o); traj_rm=res.distY_time_all;
    o=base; o.maxiters_values=Nv; o.learning_style='prm'; o.switch_eps=11;
    rng(12345); res=df.stages.run_stage_ii(cfg,o); traj_pr=res.distY_time_all;
    for a=1:size(arms,1)
        nm=arms{a,1}; sw=arms{a,2}; sty=arms{a,3};
        o=base; o.maxiters_values=Nv; o.learning_style=sty; o.switch_eps=sw;
        if strcmp(sty,'rm'), o.precomputed_distY=traj_rm; else, o.precomputed_distY=traj_pr; end
        rng(12345); r=df.stages.run_stage_ii(cfg,o);
        [ar,sh,hb]=region_area(r,1,IDTOL);
        fprintf('N=%6dk %-12s area=%.4f share=%.4f hb=%d\n', Nv/1000, nm, ar, sh, hb);
        N_col(end+1)=Nv; arm_col{end+1}=nm; area_col(end+1)=ar; share_col(end+1)=sh; hb_col(end+1)=hb; %#ok<SAGROW>
    end
    save(out,'N_col','arm_col','area_col','share_col','hb_col','Ns','IDTOL','-v7.3');   % incremental per N
end
summary=table(N_col(:), arm_col(:), area_col(:), share_col(:), logical(hb_col(:)), ...
    'VariableNames',{'N','arm','area','share','hit_boundary'});
fprintf('\n=== high-prob vs Markov area-vs-N (40x40, L1) ===\n'); disp(summary);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'hp_vs_N.csv'));
fprintf('Artifact: %s\n========== hp-vs-N complete ==========\n', out);

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
