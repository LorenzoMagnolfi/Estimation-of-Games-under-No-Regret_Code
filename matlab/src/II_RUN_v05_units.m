%% II_RUN_v05_units  Persistent types in units: predetermined against endogenous durations (R1.4.e, Q6; the lab version of the Swappa obstacle).
%  Canonical lab, N in {1M, 4M}, 61x61 grid, L1 consistency, certified tail (code 14). Both sellers run per-type Hedge copies
%  carried across units (df.sim.learn_hedge_units); a unit holds one cost pair for its duration.
%  DGPs: 'iid' (types redrawn every period, composition at N: the Section 3 reference); 'geometric' durations with mean 5
%  and mean 50, drawn before play and independent of costs (predetermined): V05's independent-unit lemma gives a valid
%  composition set with N replaced by J_eff = (sum L_j)^2 / sum L_j^2, passed through opts.cons_N_override (the consistency
%  radius and the tail's latent-count ceiling both use it); 'sale': a unit ends at the first sale by either seller, so
%  durations depend on costs (low-cost pairs price low, sell fast and are underweighted in the pooled data); the lemma's
%  condition fails, and the region is computed with the same J_eff rule to show what the mis-application does.
%  Reports J, J_eff, mean duration, the L1 distance between the realized duration-weighted type weights and the prior
%  against the composition radius, area, share, truth feasibility, regrets. Incremental saves; CSV; no figures.
clear; clc; paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
IDTOL=1e-8; NGRID=60; Ns=[1000000 4000000];
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','use_parfor',true,'fixed_gridparamV',fgV,'fixed_gridparamM',fgM,'learning_style','hedge','switch_eps',14);
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
ncpu=str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu>1 && exist('gcp','file')==2 && isempty(gcp('nocreate')), try, parpool('Processes',min(ncpu,16)); catch, end, end
dgps={'iid',NaN;'geometric',5;'geometric',50;'sale',NaN};
out=fullfile(paths.artifacts,'v05_units.mat'); S=table(); VVs=struct(); Us=struct();
for di=1:size(dgps,1)
    rule=dgps{di,1}; prm=dgps{di,2};
    for N=Ns
        rng(12345); t0=tic;
        if strcmp(rule,'iid')
            [dY,~,fr]=df.sim.learn_hedge(cfg,1,N,N,1,1,1,1); U=struct('J',N,'J_eff',N,'mean_L',1,'L1',NaN);
        else
            [dY,fr,U]=df.sim.learn_hedge_units(cfg,N,rule,prm);
        end
        tl=toc(t0);
        r_C=df.solvers.compute_consistency_slack(s^2,0.005,U.J_eff,'L1');
        o=base; o.maxiters_values=N; o.precomputed_distY={dY}; o.cons_N_override=U.J_eff;
        rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); ts=toc(t0);
        [area,share,hb,truth_in]=region_area(r,1,IDTOL);
        tag=regexprep(sprintf('%s_%g_N%d',rule,prm,N/1000),'[^A-Za-z0-9_]','_');
        S=[S; table(string(rule),prm,N,U.J,U.J_eff,U.mean_L,U.L1,r_C,area,share,hb,truth_in,max(fr(:)),tl,ts, ...
            'VariableNames',{'dur_rule','dur_param','N','J','J_eff','mean_L','L1_weights_vs_prior','r_C','area','share','hit_boundary','truth_feasible','max_realized_regret','learn_seconds','solve_seconds'})]; %#ok<AGROW>
        VVs.(tag)=r.VV_all(1,:); Us.(tag)=U;
        fprintf('%-9s param=%g N=%d: J=%d J_eff=%.0f meanL=%.2f L1=%.4f r_C=%.4f area=%.4f share=%.4f hb=%d truth=%d regret=%.3e [learn %.0fs, solve %.0fs]\n', ...
            rule,prm,N,U.J,U.J_eff,U.mean_L,U.L1,r_C,area,share,hb,truth_in,max(fr(:)),tl,ts);
        if strcmp(rule,'sale'), disp('   mean duration by (player 1 type, player 2 type):'); disp(round(U.mean_L_by_pair,2)); end
        save(out,'S','VVs','Us','dgps','Ns','fgM','fgV','IDTOL','-v7.3');
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(S,fullfile(paths.tables_ii,'v05_units.csv'));
fprintf('Artifact: %s\n========== V05 units complete ==========\n',out);

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
