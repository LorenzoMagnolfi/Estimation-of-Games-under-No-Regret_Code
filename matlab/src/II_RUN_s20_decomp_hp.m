%% II_RUN_s20_decomp_hp  Attribute the s=20 blowup, and test the high-prob fix.
%  At s=5 the consistency set contributes no area (looseness_decomp: cons_only=0),
%  so the width is all regret radius. At s=20 the formal region covers ~50% of the
%  grid and CLT does not rescue it. This run decomposes s=20 the same four ways AND
%  re-solves under the high-probability radius, on ONE learned RM trajectory:
%    exact        : eps=0,            exact consistency   (point-ID benchmark)
%    obed_Markov  : eps at aR (sw10), exact consistency   (regret radius alone)
%    cons_L1      : eps=0,            L1 consistency       (consistency alone)
%    full_Markov  : eps at aR (sw10), L1 consistency       (formal region, Markov)
%    obed_HighProb: eps at aR (sw12), exact consistency   (regret radius, high-prob)
%    full_HighProb: eps at aR (sw12), L1 consistency       (formal region, high-prob)
%  If cons_L1 ~ 0 the blowup is obedience-driven and the high-prob rows should
%  collapse it; if cons_L1 is large the blowup is genuinely the consistency object.
%  s=20, 30x30 grid, N=4M. Incremental save per region. NO figures.
clear; clc;
paths=df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=20;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; aR=0.045; aC=0.005; NGRID=30;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',4000000,'alpha_set',0.05,'backend','fast','adaptive',false, ...
    'require_corrected',true,'learning_style','rm','fixed_gridparamV',fgV,'fixed_gridparamM',fgM);
out=fullfile(paths.artifacts,'s20_decomp_hp.mat');

% full_Markov learns the trajectory; all others reuse it
o=base; o.switch_eps=10; o.alpha_R=aR; o.alpha_C=aC; o.consistency_slack_kind='L1';
rng(12345); res_fullMk=df.stages.run_stage_ii(cfg,o); traj=res_fullMk.distY_time_all;

specs={ ...
  'exact',        struct('switch_eps',10,'eps_override',0,'consistency_slack_kind','none'); ...
  'obed_Markov',  struct('switch_eps',10,'alpha_R',aR,    'consistency_slack_kind','none'); ...
  'cons_L1',      struct('switch_eps',10,'eps_override',0,'alpha_C',aC,'consistency_slack_kind','L1'); ...
  'obed_HighProb',struct('switch_eps',12,'alpha_R',aR,    'consistency_slack_kind','none'); ...
  'full_HighProb',struct('switch_eps',12,'alpha_R',aR,'alpha_C',aC,'consistency_slack_kind','L1') };

names={'full_Markov'}; R={res_fullMk};
[a,sh,hb]=region_area(res_fullMk,1,IDTOL);
area=a; share=sh; hbv=hb;
fprintf('%-14s area=%.4f share=%.4f hb=%d\n','full_Markov',a,sh,hb);
for i=1:size(specs,1)
    nm=specs{i,1}; o=base; f=fieldnames(specs{i,2});
    for j=1:numel(f), o.(f{j})=specs{i,2}.(f{j}); end
    o.precomputed_distY=traj; rng(12345);
    res=df.stages.run_stage_ii(cfg,o);
    [a,sh,hb]=region_area(res,1,IDTOL);
    fprintf('%-14s area=%.4f share=%.4f hb=%d\n',nm,a,sh,hb);
    names{end+1}=nm; R{end+1}=res; area(end+1)=a; share(end+1)=sh; hbv(end+1)=hb; %#ok<SAGROW>
    save(out,'names','area','share','hbv','IDTOL','-v7.3');   % incremental
end
summary=table(names(:), area(:), share(:), logical(hbv(:)), ...
    'VariableNames',{'region','area','share','hit_boundary'});
fprintf('\n=== s=20 decomposition + high-prob (N=4M, 30x30) ===\n'); disp(summary);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'s20_decomp_hp.csv'));
save(out,'names','area','share','hbv','R','IDTOL','-v7.3');
fprintf('Artifact: %s\n========== s=20 decomp+HP complete ==========\n', out);

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
