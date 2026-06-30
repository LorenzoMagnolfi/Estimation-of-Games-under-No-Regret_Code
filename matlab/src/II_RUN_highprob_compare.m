%% II_RUN_highprob_compare  (Niccolo Proposal 3: high-probability vs Markov envelopes)
%  The Markov step turns E[R]~1/sqrt(N) into a 1/alpha tail; a high-probability bound
%  gives sqrt(ln(1/alpha)) instead. Compare the identified region under the Markov
%  radius (switch_eps 10/11) and the high-probability radius (12/13), for both full
%  feedback (RM) and bandit (PRM), all at L1 consistency, N=4M, 60x60 grid. Key
%  question: does the high-probability bandit radius become informative? Learns one
%  RM and one PRM trajectory and re-solves. NO figures.
%  NOTE: the 12/13 HP constants are ILLUSTRATIVE (TBD with NL); this run shows the
%  ORDER of the tightening, not a final radius.
clear; clc;
paths = df_repo_paths(); rng(12345);
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg = df.setup.game_simulation(NPlayers, alpha, actions_vec, mu, sigma2, s);
IDTOL=1e-8; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',4000000,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005, ...
    'backend','fast','adaptive',false,'require_corrected',true,'consistency_slack_kind','L1', ...
    'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);

o=base; o.learning_style='rm'; o.switch_eps=10; rng(12345); res_rm_mk=df.stages.run_stage_ii(cfg,o); traj_rm=res_rm_mk.distY_time_all;
o=base; o.learning_style='rm'; o.switch_eps=12; o.precomputed_distY=traj_rm; rng(12345); res_rm_hp=df.stages.run_stage_ii(cfg,o);
o=base; o.learning_style='prm'; o.switch_eps=11; rng(12345); res_pr_mk=df.stages.run_stage_ii(cfg,o); traj_pr=res_pr_mk.distY_time_all;
o=base; o.learning_style='prm'; o.switch_eps=13; o.precomputed_distY=traj_pr; rng(12345); res_pr_hp=df.stages.run_stage_ii(cfg,o);

names={'RM Markov(10)','RM HighProb(12)','PRM Markov(11)','PRM HighProb(13)'};
R={res_rm_mk,res_rm_hp,res_pr_mk,res_pr_hp};
area=zeros(4,1); share=zeros(4,1); hb=false(4,1);
for k=1:4, [area(k),share(k),hb(k)]=region_area(R{k},1,IDTOL); end
summary=table(names(:),area,share,hb,'VariableNames',{'radius','area','share','hit_boundary'});
fprintf('\n=== Prop 3: high-probability vs Markov radius (N=4M, L1) ===\n'); disp(summary);

if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'highprob_compare.csv'));
save(fullfile(paths.artifacts,'highprob_compare.mat'),'res_rm_mk','res_rm_hp','res_pr_mk','res_pr_hp','summary','IDTOL','-v7.3');
fprintf('Artifact: %s\n========== high-prob compare complete ==========\n', fullfile(paths.artifacts,'highprob_compare.mat'));

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
