%% II_RUN_v05_nonparam  Nonparametric cost-distribution clouds under the certified calibrations (R1.1.b).
%  The candidate is a probability vector on the type support (true + 1,000 local + 9,000 global + 200 spiky = 10,201
%  candidates, as in July). s = 5: N in {0.5, 1, 2, 4}M with Hedge tailF (14), Hedge momentF (16) and EXP3-IX tailB (15),
%  same trajectory per learner and N. s = 10: N in {1, 4}M with Hedge tailF only (the cone is four times larger).
%  Reports identified counts and shares, the truth candidate's feasibility, and the (mean, variance) projection of the
%  identified cloud. July's runs used regret matching and the legacy radius (no M_R). Preflight on a tiny cloud first.
%  Incremental saves; CSV; no figures.
clear; clc; paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2);
IDTOL=1e-8;
base=struct('alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','consistency_slack_kind','L1','use_parfor',true);

%% preflight: tiny cloud, small N, both learners, three codes; VV finite and the truth candidate identified
cfg5=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,5);
pre=base; pre.maxiters_values=40000; pre.K_local=20; pre.K_global=20; pre.K_spiky=5;
rng(7); [dYp,~,~]=df.sim.learn_hedge(cfg5,1,40000,40000,1,1,1,1);
for sw=[14 16]
    o=pre; o.learning_style='hedge'; o.switch_eps=sw; o.precomputed_distY={dYp}; r=df.stages.run_stage_ii_nonparam(cfg5,o);
    assert(all(isfinite(r.VV_all(1,:))),'preflight: non-finite VV'); assert(r.VV_all(1,1)<=IDTOL,'preflight: truth not identified (sw%d)',sw);
    fprintf('preflight hedge sw%d ok: %d/%d identified\n',sw,nnz(r.VV_all(1,:)<=IDTOL),size(r.VV_all,2));
end
rng(7); [dYx,~,~]=df.sim.learn_exp3ix(cfg5,1,40000,40000,1,1,1,1);
o=pre; o.learning_style='exp3ix'; o.switch_eps=15; o.precomputed_distY={dYx}; r=df.stages.run_stage_ii_nonparam(cfg5,o);
assert(all(isfinite(r.VV_all(1,:))) && r.VV_all(1,1)<=IDTOL,'preflight: exp3ix tailB failed');
fprintf('preflight exp3ix sw15 ok: %d/%d identified\nPREFLIGHT_OK\n',nnz(r.VV_all(1,:)<=IDTOL),size(r.VV_all,2));

%% production
out=fullfile(paths.artifacts,'v05_nonparam.mat'); R=table(); clouds=struct();
plans={5,[500000 1000000 2000000 4000000],{'hedge',14;'hedge',16;'exp3ix',15}; 10,[1000000 4000000],{'hedge',14}};
for pi_=1:size(plans,1)
    s=plans{pi_,1}; Ns=plans{pi_,2}; arms=plans{pi_,3};
    cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
    for Nv=Ns
        traj=struct();
        rng(12345); t0=tic; [dY_h,~,fr_h]=df.sim.learn_hedge(cfg,1,Nv,Nv,1,1,1,1); traj.hedge={dY_h}; fprintf('s=%d N=%dk hedge learned %.0fs\n',s,Nv/1000,toc(t0));
        if any(strcmp(arms(:,1),'exp3ix'))
            rng(12345); t0=tic; [dY_x,~,fr_x]=df.sim.learn_exp3ix(cfg,1,Nv,Nv,1,1,1,1); traj.exp3ix={dY_x}; fprintf('s=%d N=%dk exp3ix learned %.0fs\n',s,Nv/1000,toc(t0));
        else
            fr_x=NaN;
        end
        for a=1:size(arms,1)
            o=base; o.maxiters_values=Nv; o.learning_style=arms{a,1}; o.switch_eps=arms{a,2}; o.precomputed_distY=traj.(arms{a,1});
            rng(12345); t0=tic; r=df.stages.run_stage_ii_nonparam(cfg,o); ts=toc(t0);
            VV=r.VV_all(1,:); id=VV<=IDTOL; dp=squeeze(r.distpars_all(1,:,:));
            cnt=nnz(id); share=cnt/numel(VV); truth_in=VV(1)<=IDTOL;
            if cnt>0, mrange=[min(dp(id,1)) max(dp(id,1))]; vrange=[min(dp(id,2)) max(dp(id,2))]; else, mrange=[NaN NaN]; vrange=[NaN NaN]; end
            mr=max(fr_h(:)); if strcmp(arms{a,1},'exp3ix'), mr=max(fr_x(:)); end
            R=[R; table(s,Nv,string(arms{a,1}),arms{a,2},cnt,numel(VV),share,truth_in,mrange(1),mrange(2),vrange(1),vrange(2),mr,ts, ...
                'VariableNames',{'s','N','learner','switch_eps','n_identified','n_candidates','share','truth_identified','mean_lo','mean_hi','var_lo','var_hi','max_realized_regret','solve_seconds'})]; %#ok<AGROW>
            key=sprintf('s%d_N%d_%s_sw%d',s,Nv/1000,arms{a,1},arms{a,2}); clouds.(key)=struct('VV',VV,'distpars',dp);
            fprintf('s=%2d N=%dk %-7s sw%d: %d/%d identified (share %.4f), truth %d, mean [%.3f, %.3f], var [%.3f, %.3f] [%.0fs]\n', ...
                s,Nv/1000,arms{a,1},arms{a,2},cnt,numel(VV),share,truth_in,mrange(1),mrange(2),vrange(1),vrange(2),ts);
            save(out,'R','clouds','IDTOL','-v7.3');
        end
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(R,fullfile(paths.tables_ii,'v05_nonparam.csv'));
fprintf('\n=== V05 nonparametric clouds ===\n'); disp(R);
fprintf('Artifact: %s\n========== V05 nonparam complete ==========\n',out);
