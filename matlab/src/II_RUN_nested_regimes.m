%% II_RUN_nested_regimes  One trajectory, progressively stronger envelopes.
%  NL memo §10.1 preferred design: generate data under a strong DGP (full
%  feedback regret matching) and build regions under progressively stronger
%  maintained assumptions on the SAME data, so nesting is interpretable:
%  the FF learner's regret satisfies the weaker bandit-class envelope a
%  fortiori, so the bandit-envelope region is a valid conservative region for
%  the same data. Arms: bandit Markov (sw11), FF Markov (sw10), bandit
%  high-prob (sw13, illustrative), FF high-prob (sw12, illustrative).
%  Envelope-ordered pairs must nest; Markov-FF vs HP-bandit is the lattice
%  (not ordered). 61x61 grid, L1 consistency, direct lane + parfor.
clear; clc;
paths=df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
IDTOL=1e-8; N=4000000; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','learning_style','rm','use_parfor',true, ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

o=base; o.switch_eps=10; o.learn_only=true;
rng(12345); res=df.stages.run_stage_ii(cfg,o); traj=res.distY_time_all;

arms=struct('name',{'bandit_markov','ff_markov','bandit_hp','ff_hp'}, ...
    'sw',{11,10,13,12});
n=numel(arms); name=cell(n,1); sw=zeros(n,1); area=zeros(n,1); share=zeros(n,1); hb=zeros(n,1);
out=fullfile(paths.artifacts,'nested_regimes.mat');
for k=1:n
    o=base; o.switch_eps=arms(k).sw; o.precomputed_distY=traj;
    rng(12345); r=df.stages.run_stage_ii(cfg,o);
    [area(k),share(k),hb(k)]=region_area(r,1,IDTOL);
    name{k}=arms(k).name; sw(k)=arms(k).sw;
    fprintf('%-14s (sw%d): area=%.4f share=%.4f boundary=%d\n', ...
        name{k}, sw(k), area(k), share(k), hb(k));
    save(out,'name','sw','area','share','hb','IDTOL','-v7.3');
end
% nesting checks on envelope-ordered pairs (same trajectory throughout)
ok = area(1) >= area(2) - 1e-9 && area(1) >= area(3) - 1e-9 && ...
     area(2) >= area(4) - 1e-9 && area(3) >= area(4) - 1e-9;
fprintf('envelope-ordered nesting holds: %s\n', string(ok));
summary=table(name,sw,area,share,hb);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'nested_regimes.csv'));
fprintf('Artifact: %s\n========== nested regimes complete ==========\n', out);

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
