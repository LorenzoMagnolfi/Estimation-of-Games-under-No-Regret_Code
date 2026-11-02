%% II_RUN_v05_realization  Realization-conditional (persistent-type) identification under the certified tail (R1.4.e).
%  Data-only specification of App. SIM-RC: for a fixed realized cost pair, the empirical action distribution must
%  satisfy each player's own regret inequality at the candidate costs (df.solvers.compute_dataonly_regret). With a
%  persistent type each player has ONE context visited every period, so the count is exactly N and no composition set
%  is needed; the obedience budget alpha = 0.05 is split over the two players, delta = 0.025 each. Radii per player and
%  candidate cost (payoff span K at the candidate):
%    tailF    K min{1, [c_F sqrt(N) + sqrt((N + 1) ln(J_N / delta))] / N}          certified direct tail (family F)
%    momentF  (2 / alpha) K (c_F + 1/4) / sqrt(N)                                  valid first-moment rectangle
%    legacy   K sqrt(2 ln|A| / N) / alpha                                           April's per-candidate radius (no union bound)
%  Learner: df.sim.learn_fixed_hedge (family F), seed 12345 per (truth, N). Truths (3,3), (2,4), (1,5) on the s = 5
%  cost grid at N in {0.25, 0.5, 1, 2, 4}M, and the same truths on the s = 20 grid at 1M and 4M. Reports set
%  cardinality, truth feasibility, and the realized regrets. Closed-form regrets, no LP; minutes. CSV; no figures.
clear; clc; paths=df_repo_paths();
NPlayers=2; alpha_d=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2);
alpha=0.05; delta=alpha/2;
plans={5,[3 3;2 4;1 5],[250000 500000 1000000 2000000 4000000]; 20,[12 12;8 16;4 20],[1000000 4000000]};
out=fullfile(paths.artifacts,'v05_realization.mat'); R=table(); feas=struct();
for pi_=1:size(plans,1)
    s=plans{pi_,1}; truths=plans{pi_,2}; Ns=plans{pi_,3};
    cfg=df.setup.game_simulation(NPlayers,alpha_d,actions_vec,mu,sigma2,s);
    d=cfg.NAct; cF=1.25*sqrt(log(d));
    K=zeros(s,NPlayers); for j=1:NPlayers, K(:,j)=(max(cfg.Pi(:,:,j),[],1)-min(cfg.Pi(:,:,j),[],1))'; end
    grid=cfg.type_space{1,1};
    for tr=1:size(truths,1)
        for Nv=Ns
            rng(12345); t0=tic; [dY,fr]=df.sim.learn_fixed_hedge(cfg,Nv,truths(tr,:)); tl=toc(t0);
            JN=1+ceil(log2(Nv));
            radii.tailF=K.*min(1,(cF*sqrt(Nv)+sqrt((Nv+1)*log(JN/delta)))/Nv);
            radii.momentF=(2/alpha)*K*(cF+0.25)/sqrt(Nv);
            radii.legacy=K*sqrt(2*log(d)/Nv)/alpha;
            R1=zeros(s,s); R2=zeros(s,s);
            for i=1:s, for j=1:s, [R1(i,j),R2(i,j)]=df.solvers.compute_dataonly_regret(cfg,dY,grid(i),grid(j)); end, end
            names=fieldnames(radii);
            for r_=1:numel(names)
                e=radii.(names{r_}); F=false(s,s);
                for i=1:s, for j=1:s, F(i,j)=(R1(i,j)<=e(i,1)) && (R2(i,j)<=e(j,2)); end, end
                truth_in=F(truths(tr,1),truths(tr,2));
                key=sprintf('s%d_t%d%d_N%d_%s',s,truths(tr,1),truths(tr,2),Nv/1000,names{r_}); feas.(key)=F;
                R=[R; table(s,string(sprintf('(%d,%d)',truths(tr,1),truths(tr,2))),Nv,string(names{r_}),nnz(F),s^2,truth_in,max(fr),tl, ...
                    'VariableNames',{'s','truth','N','radius','n_feasible','n_candidates','truth_feasible','max_realized_regret','learn_seconds'})]; %#ok<AGROW>
                fprintf('s=%2d truth (%d,%d) N=%dk %-8s: %d/%d feasible, truth %d, realized regret %.3e\n',s,truths(tr,1),truths(tr,2),Nv/1000,names{r_},nnz(F),s^2,truth_in,max(fr));
            end
            save(out,'R','feas','alpha','delta','-v7.3');
        end
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(R,fullfile(paths.tables_ii,'v05_realization.csv'));
fprintf('\n=== V05 realization-conditional sets ===\n'); disp(R);
fprintf('Artifact: %s\n========== V05 realization complete ==========\n',out);
