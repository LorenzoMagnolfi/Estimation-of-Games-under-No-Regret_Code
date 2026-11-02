%% II_RUN_v05_demand_c3  Joint demand and cost on a wide two-stage cost grid, and the C3 projection with a first-stage demand set (R1.1.d; plan item 5).
%  One Hedge trajectory at N = 4M under the true slope eta_0 = -1/3 (seed 12345). Three objects:
%   (A) known-payoff cost sets at each of 39 slopes in [-1, -0.05] (plus eta_0) at alpha_R = 0.045, alpha_C = 0.005, on a
%       WIDE 31 x 31 window (mu in [1.5, 7.5], sigma^2 in [0.15, 3.5]; the 2 November 31 x 31 run stopped at mu = 4.5 and
%       truncated the demand-cost ridge) followed by a LOCAL 21 x 21 window (mu +- 0.3, sigma^2 +- 0.35) around the feasible
%       cells' centroid or, if none, the least-infeasible cell. The union over slopes is the demand-free projection
%       (maintained domain, alpha_D = 0, V05 eq. s4projection).
%   (B) the C3 projection: a first-stage confidence interval for eta from n_D = 10,000 simulated sales (profiles drawn from
%       the empirical profile distribution, one logit sale draw each at eta_0; MLE and observed information) at level
%       alpha_D = 0.01; the cost set at each of 9 slopes spanning the interval at alpha_R = 0.035 (alpha_L = 0.04); the union
%       is the C3 set. Same data, no independence required.
%  Direct SeDuMi lane with parfor. Incremental saves; CSV; no figures.
clear; clc; paths=df_repo_paths();
lab=struct('actions_vec',[4;5;6;7;8],'mu_true',[3;3],'sigma2_true',eye(2),'s_val',5,'IDTOL',1e-8);
eta_true=-1/3; T=4000000; alpha_C=0.005; alpha_D=0.01; n_D=10000;
etaA=unique([linspace(-1.0,-0.05,39), eta_true]);
cfg_true=df.setup.game_simulation(2,eta_true,lab.actions_vec,lab.mu_true,lab.sigma2_true,lab.s_val);
rng(12345); t0=tic; [dY,~,fr]=df.sim.learn_hedge(cfg_true,1,T,T,1,1,1,1);
fprintf('[DGP] Hedge learned %.0fs, max realized regret %.3e\n',toc(t0),max(fr(:)));
action_distribution=dY;
%% first stage: simulated sales and the MLE of the slope
rng(777); k=randsample((1:size(cfg_true.A,1))',n_D,true,dY(:,1)); P=cfg_true.A(k,:);
e1=exp(eta_true*P(:,1)); e2=exp(eta_true*P(:,2)); D=1+e1+e2; u=rand(n_D,1);
y=zeros(n_D,1); y(u<e1./D)=1; y(u>=e1./D & u<(e1+e2)./D)=2;
nll=@(eta) -sum((y==1).*(eta*P(:,1))+(y==2).*(eta*P(:,2))-log(1+exp(eta*P(:,1))+exp(eta*P(:,2))));
eta_hat=fminbnd(nll,-3,0); h=1e-4; info_obs=(nll(eta_hat+h)-2*nll(eta_hat)+nll(eta_hat-h))/h^2; se=1/sqrt(info_obs);
z=-norminv(alpha_D/2); ci=[eta_hat-z*se, eta_hat+z*se];
fprintf('[first stage] n_D=%d sales (%d seller 1, %d seller 2, %d none): eta_hat=%.4f se=%.4f, %.0f-percent CI [%.4f, %.4f] (truth %.4f)\n', ...
    n_D,nnz(y==1),nnz(y==2),nnz(y==0),eta_hat,se,100*(1-alpha_D),ci(1),ci(2),eta_true);
etaB=linspace(ci(1),ci(2),9);
%% per-slope two-stage cost sets
ncpu=str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu>1 && exist('gcp','file')==2 && isempty(gcp('nocreate')), try, parpool('Processes',min(ncpu,16)); catch, end, end
wideM=linspace(0.5,2.5,31)'; wideV=linspace(0.15,3.5,31)';     % multipliers of mu_true(1) = 3 and sigma2_true(1,1) = 1
sets={'A',etaA,0.045;'B',etaB,0.035};
edgestr={'',' EDGE'};
out=fullfile(paths.artifacts,'v05_demand_c3.mat'); R=table(); W=struct(); Lc=struct();
for si=1:size(sets,1)
    sname=sets{si,1}; etas=sets{si,2}; aR=sets{si,3};
    for ei=1:numel(etas)
        eta_c=etas(ei); t0=tic;
        [VVw,dpw]=v05_solve_eta(eta_c,[1;wideM],[1;wideV],T,aR,alpha_C,14,action_distribution,lab);
        fw=VVw<=lab.IDTOL; nw=nnz(fw);
        if nw>0, ctr=[mean(dpw(fw,1)),mean(dpw(fw,2))]; else, [~,im]=min(VVw); ctr=dpw(im,:); end
        locM=linspace(max(ctr(1)-0.3,0.5),ctr(1)+0.3,21)'/lab.mu_true(1);
        locV=linspace(max(ctr(2)-0.35,0.05),ctr(2)+0.35,21)'/lab.sigma2_true(1,1);
        [VVl,dpl]=v05_solve_eta(eta_c,[1;locM],[1;locV],T,aR,alpha_C,14,action_distribution,lab);
        fl=VVl<=lab.IDTOL; nl=nnz(fl);
        edge=nw>0 && (any(abs(dpw(fw,1)-7.5)<1e-6) || any(abs(dpw(fw,2)-3.5)<1e-6) || any(abs(dpw(fw,1)-1.5)<1e-6) || any(abs(dpw(fw,2)-0.15)<1e-6));
        if nl>0, box=[min(dpl(fl,1)),max(dpl(fl,1)),min(dpl(fl,2)),max(dpl(fl,2))]; else, box=nan(1,4); end
        R=[R; table(string(sname),eta_c,aR,nw,min(VVw),ctr(1),ctr(2),edge,nl,box(1),box(2),box(3),box(4),VVl(1)<=lab.IDTOL,toc(t0), ...
            'VariableNames',{'set','eta','alpha_R','n_feasible_wide','min_VV_wide','center_mu','center_sigma2','wide_hits_edge','n_feasible_local','mu_lo','mu_hi','sigma2_lo','sigma2_hi','truth_cell_feasible','seconds'})]; %#ok<AGROW>
        tag=sprintf('%s_%d',sname,ei); W.(tag)=struct('eta',eta_c,'VV',VVw,'dp',dpw); Lc.(tag)=struct('eta',eta_c,'VV',VVl,'dp',dpl);
        fprintf('[%s %2d/%d] eta=%.4f aR=%.3f: wide %d feasible (minVV %.2e, center (%.2f, %.2f)%s); local %d feasible, mu [%.3f, %.3f] sigma2 [%.3f, %.3f]; truth cell %d [%.0fs]\n', ...
            sname,ei,numel(etas),eta_c,aR,nw,min(VVw),ctr(1),ctr(2),edgestr{edge+1},nl,box(1),box(2),box(3),box(4),VVl(1)<=lab.IDTOL,toc(t0));
        save(out,'R','W','Lc','eta_true','eta_hat','se','ci','etaA','etaB','T','alpha_C','alpha_D','n_D','lab','-v7.3');
    end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(R,fullfile(paths.tables_ii,'v05_demand_c3.csv'));
for si=1:size(sets,1)
    sname=sets{si,1}; rows=R(R.set==sname & R.n_feasible_local>0,:);
    if isempty(rows), fprintf('%s: no feasible cell at any slope\n',sname); continue; end
    fprintf('%s: slopes with a feasible cost cell: [%.4f, %.4f] (%d of %d); projection of the union: mu [%.3f, %.3f], sigma2 [%.3f, %.3f]; wide-window edge hits %d\n', ...
        sname,min(rows.eta),max(rows.eta),height(rows),nnz(R.set==sname),min(rows.mu_lo),max(rows.mu_hi),min(rows.sigma2_lo),max(rows.sigma2_hi),nnz(rows.wide_hits_edge));
end
rt=R(abs(R.eta-eta_true)<1e-9 & R.set=="A",:);
if ~isempty(rt), fprintf('known slope (eta_0): %d local cells, mu [%.3f, %.3f], sigma2 [%.3f, %.3f]\n',rt.n_feasible_local,rt.mu_lo,rt.mu_hi,rt.sigma2_lo,rt.sigma2_hi); end
fprintf('Artifact: %s\n========== V05 demand C3 complete ==========\n',out);
