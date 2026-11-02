%% II_RUN_v05_mixed  SIM-MIXED: one Hedge learner against a non-learning rival (R1.2, R1.4.f).
%  Canonical lab (two sellers, five prices, s = 5), N in {1M, 4M}, 61x61 grid, L1 consistency, certified tail (code 14).
%  DGPs: (a) Hedge vs Hedge (both learners, the Section 3 frontier); (b) Hedge vs a constant price of 6 whatever the
%  rival's cost; (c) Hedge vs a myopic best-responder to the learner's previous price. For each DGP two regions:
%  'joint' imposes both players' obedience rows with the family-F tolerance (mis-specified for the non-learner); 'own'
%  imposes the learner's rows only (the rival's rows are relaxed to the payoff span, opts.eps_player_cap = 2). Reports
%  area, share, truth feasibility, the realized positive-part regrets of both players, and the rival's action
%  frequencies. The claim to test: the learner's own set is valid whether or not the rival learns; the joint set is not.
%  Direct SeDuMi lane with parfor. Incremental saves; CSV; no figures.
clear; clc; paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
IDTOL=1e-8; NGRID=60; Ns=[1000000 4000000];
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','use_parfor',true,'fixed_gridparamV',fgV,'fixed_gridparamM',fgM,'learning_style','hedge','switch_eps',14);
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
ncpu=str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu>1 && exist('gcp','file')==2 && isempty(gcp('nocreate')), try, parpool('Processes',min(ncpu,16)); catch, end, end
dgps={'hedge_vs_hedge','',NaN;'hedge_vs_constant6','constant',6;'hedge_vs_myopic','myopic',NaN};
out=fullfile(paths.artifacts,'v05_mixed.mat'); S=table(); VVs=struct(); oppfreqs=struct();
for di=1:size(dgps,1)
    name=dgps{di,1}; rule=dgps{di,2}; prm=dgps{di,3};
    for N=Ns
        rng(12345); t0=tic;
        if isempty(rule)
            [dY,~,fr]=df.sim.learn_hedge(cfg,1,N,N,1,1,1,1); oppfreq=nan(numel(actions_vec),1);
        else
            [dY,fr,info]=df.sim.learn_mixed(cfg,1,N,rule,prm); oppfreq=info.opp_action_freq;
        end
        tl=toc(t0);
        oppfreqs.(sprintf('%s_N%d',name,N/1000))=oppfreq;
        for spec={'joint','own'}
            o=base; o.maxiters_values=N; o.precomputed_distY={dY};
            if strcmp(spec{1},'own'), o.eps_player_cap=2; end
            rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); ts=toc(t0);
            [area,share,hb,truth_in]=region_area(r,1,IDTOL);
            S=[S; table(string(name),string(spec{1}),N,area,share,hb,truth_in,max(fr(:,1)),max(fr(:,2)),tl,ts, ...
                'VariableNames',{'dgp','spec','N','area','share','hit_boundary','truth_feasible','regret_learner','regret_rival','learn_seconds','solve_seconds'})]; %#ok<AGROW>
            VVs.(sprintf('%s_%s_N%d',name,spec{1},N/1000))=r.VV_all(1,:);
            fprintf('%-18s %-5s N=%d: area=%.4f share=%.4f hb=%d truth=%d regret learner=%.3e rival=%.3e [learn %.0fs, solve %.0fs]\n', ...
                name,spec{1},N,area,share,hb,truth_in,max(fr(:,1)),max(fr(:,2)),tl,ts);
            save(out,'S','VVs','oppfreqs','dgps','Ns','fgM','fgV','IDTOL','-v7.3');
        end
        fprintf('   rival action frequencies over the price grid: %s\n',mat2str(oppfreq',3));
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(S,fullfile(paths.tables_ii,'v05_mixed.csv'));
fprintf('Artifact: %s\n========== V05 mixed complete ==========\n',out);

function [area,share,hb,truth_in]=region_area(res,ni,IDTOL)
    nV=res.NGridV+1; nM=res.NGridM+1;
    dp=squeeze(res.distpars_all(ni,:,:));
    mu_g=reshape(dp(:,1),nV,nM); mu_g=mu_g(2:end,2:end);
    sg_g=reshape(dp(:,2),nV,nM); sg_g=sg_g(2:end,2:end);
    VV_g=reshape(res.VV_all(ni,:),nV,nM); VV_g=VV_g(2:end,2:end);
    id=VV_g<=IDTOL; n_id=nnz(id);
    dmu=median(diff(unique(mu_g(:))),'omitnan'); dsg=median(diff(unique(sg_g(:))),'omitnan');
    area=n_id*abs(dmu)*abs(dsg); share=n_id/numel(VV_g);
    hb=any(id(1,:))||any(id(end,:))||any(id(:,1))||any(id(:,end));
    truth_in=res.VV_all(ni,1)<=IDTOL;
end
