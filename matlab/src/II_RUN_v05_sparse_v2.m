%% II_RUN_v05_sparse_v2  Sparse-outcome exhibit, rerun after the learn_hedge scaling fix of 4 November 2026 (finding I4): what realized-sales feedback costs, as the sale probability falls.
%  The canonical lab with the sale probability scaled by q in {1, 0.3, 0.1, 0.03}: structural payoff u = q cp (p - c)
%  (cfg.Pi scaled by q), realized profit 1{sale} (p - c) with sale ~ Bernoulli(q cp). The identified set in (mu, sigma^2)
%  is invariant to q (the obedience inequalities are homogeneous in q), so differences across q are entirely the
%  certified allowance. Three learners at N = 4M on the 61x61 grid, L1 consistency:
%    hedge        expected profits observed (family F), tailF (14): the benchmark; its tolerance scales with K ~ q
%    hedge_noisy  full potential-profit vector observed, Hedge on Y (family F with span L), code 18 (tail + bridge)
%    exp3ix_noisy realized own profit only, EXP3-IX on Y (family B with width L), code 19 (tail + bridge)
%  L is the realized-profit range in cash units and does not shrink with q, so the bridge-to-cap ratio grows as 1/q:
%  the Swappa mechanism in the lab. Reports area, share, truth feasibility, realized structural regret, and the ratio of
%  the certified tolerance to the cap at the truth. Incremental saves; CSV; no figures.
clear; clc; paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
IDTOL=1e-8; N=4000000; NGRID=60;
fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false, ...
    'require_corrected',true,'consistency_slack_kind','L1','use_parfor',true,'fixed_gridparamV',fgV,'fixed_gridparamM',fgM);
qs=[1 0.3 0.1 0.03];
arms={'hedge',14; 'hedge_noisy',18; 'exp3ix_noisy',19};
out=fullfile(paths.artifacts,'v05_sparse_v2.mat'); S=table(); VVs=struct();
for qi=1:numel(qs)
    q=qs(qi);
    cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
    cfg.Pi=q*cfg.Pi; cfg.sale_scale=q;
    phi=cfg.marg_distrib(:)'; r_L1=df.solvers.compute_consistency_slack(s^2,0.005,N,'L1');
    Kt=max(cfg.Pi(:,:,1))-min(cfg.Pi(:,:,1));
    for a=1:size(arms,1)
        sty=arms{a,1}; sw=arms{a,2};
        rng(12345); t0=tic;
        switch sty
            case 'hedge',        [dY,~,fr]=df.sim.learn_hedge(cfg,1,N,N,1,1,1,1);
            case 'hedge_noisy',  [dY,~,fr]=df.sim.learn_hedge_noisy(cfg,1,N,N,1,1,1,1);
            case 'exp3ix_noisy', [dY,~,fr]=df.sim.learn_exp3ix_noisy(cfg,1,N,N,1,1,1,1);
        end
        tl=toc(t0);
        eps_truth=df.solvers.compute_epsilon_v05(cfg,N,phi,r_L1,sw,0.045);     % per-type tolerance at the truth
        tol_over_cap=eps_truth./(Kt.*phi);                                        % 1 means the tolerance equals the cap
        o=base; o.learning_style=sty; o.switch_eps=sw; o.precomputed_distY={dY};
        rng(12345); t0=tic; r=df.stages.run_stage_ii(cfg,o); ts=toc(t0);
        [area,share,hb,truth_in]=region_area(r,1,IDTOL);
        S=[S; table(q,string(sty),sw,area,share,hb,truth_in,max(fr(:)),max(tol_over_cap),median(tol_over_cap),tl,ts, ...
            'VariableNames',{'q','learner','switch_eps','area','share','hit_boundary','truth_feasible','max_realized_regret','tol_over_cap_max','tol_over_cap_median','learn_seconds','solve_seconds'})]; %#ok<AGROW>
        VVs.(sprintf('q%03d_%s',round(1000*q),sty))=r.VV_all(1,:);
        fprintf('q=%.2f %-12s sw%d: area=%.4f share=%.4f hb=%d truth=%d regret=%.3e tol/cap max=%.3f med=%.3f [learn %.0fs, solve %.0fs]\n', ...
            q,sty,sw,area,share,hb,truth_in,max(fr(:)),max(tol_over_cap),median(tol_over_cap),tl,ts);
        save(out,'S','VVs','qs','IDTOL','N','-v7.3');
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(S,fullfile(paths.tables_ii,'v05_sparse_v2.csv'));
fprintf('\n=== V05 sparse-outcome exhibit (N=4M, 61x61) ===\n'); disp(S);
fprintf('Artifact: %s\n========== V05 sparse complete ==========\n',out);

function [area,share,hb,truth_in]=region_area(res,ni,IDTOL)
    nV=res.NGridV+1; nM=res.NGridM+1;
    dp=squeeze(res.distpars_all(ni,:,:));
    mu_g=reshape(dp(:,1),nV,nM); mu_g=mu_g(2:end,2:end);
    sg_g=reshape(dp(:,2),nV,nM); sg_g=sg_g(2:end,2:end);
    VVfull=reshape(res.VV_all(ni,:),nV,nM); truth_in=VVfull(1,1)<=IDTOL;
    VV=VVfull(2:end,2:end); id=VV<=IDTOL;
    share=nnz(id)/numel(id);
    dmu=median(diff(unique(mu_g(:))),'omitnan'); dsg=median(diff(unique(sg_g(:))),'omitnan');
    area=nnz(id)*abs(dmu)*abs(dsg);
    hb=any(id(1,:))||any(id(end,:))||any(id(:,1))||any(id(:,end));
end
