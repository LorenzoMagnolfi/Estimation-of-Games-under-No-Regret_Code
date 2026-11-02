%% II_SMOKE_v05b  Smoke test for the batch 7 pieces (learn_mixed, learn_hedge_units, eps_player_cap, cons_N_override, v05_solve_eta).
%  Tiny lab (N = 20k, 4x4 grid), minutes. Checks: (0) lint of the new files; (1) learn_mixed runs under both rival rules,
%  distY sums to one, regrets finite, the constant rival posts its price; (2) learn_hedge_units runs under both duration
%  rules with durations summing to N, J_eff <= J, finite weights and regrets; (3) relaxing the rival's obedience rows
%  (eps_player_cap = 2) weakly enlarges the region cell by cell; (4) a smaller composition count (cons_N_override = J_eff)
%  weakly enlarges the region cell by cell; (5) the first-stage MLE recovers the slope and v05_solve_eta returns finite VV
%  on a wide and a local grid. Run: matlab -batch "II_CHECK_corrected_bundle_prereqs; II_SMOKE_v05b"
%  Prints SMOKE_V05B: PASS|FAIL and exits nonzero on FAIL.
fprintf('=== SMOKE_V05B_START ===\n'); ok=true;
files={fullfile('+df','+sim','learn_mixed.m'),fullfile('+df','+sim','learn_hedge_units.m'),fullfile('+df','+stages','run_stage_ii.m'), ...
    'v05_solve_eta.m','II_RUN_v05_mixed.m','II_RUN_v05_units.m','II_RUN_v05_demand_c3.m'};
for i=1:numel(files)
    m=checkcode(files{i},'-string');
    if isempty(strtrim(m)), fprintf('CHECKCODE_CLEAN %s\n',files{i}); else, fprintf('CHECKCODE %s:\n%s\n',files{i},m); end
end
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5; N=20000;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
NGRID=4; IDTOL=1e-8; fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
base=struct('maxiters_values',N,'alpha_set',0.05,'alpha_R',0.045,'alpha_C',0.005,'backend','fast','adaptive',false,'require_corrected',true, ...
    'consistency_slack_kind','L1','use_parfor',false,'fixed_gridparamV',fgV,'fixed_gridparamM',fgM,'learning_style','hedge','switch_eps',14);
%% (1) learn_mixed
dYc=[]; dYg=[];
try
    rng(1); [dYc,frc,infc]=df.sim.learn_mixed(cfg,1,N,'constant',6);
    rng(1); [dYm,frm,~]=df.sim.learn_mixed(cfg,1,N,'myopic',NaN);
    c1=abs(sum(dYc)-1)<1e-9 && abs(sum(dYm)-1)<1e-9 && all(isfinite(frc(:))) && all(isfinite(frm(:))) && all(frc(:)>=0);
    c2=abs(infc.opp_action_freq(3)-1)<2/N;   % price 6 is the third action; only the uniform first period differs
    fprintf('(1) learn_mixed: sums ok=%d; constant rival posts 6 with frequency %.5f; learner regret max %.3e (vs constant) %.3e (vs myopic); rival regret max %.3e / %.3e\n', ...
        c1,infc.opp_action_freq(3),max(frc(:,1)),max(frm(:,1)),max(frc(:,2)),max(frm(:,2)));
    ok=ok&&c1&&c2;
catch err, fprintf('(1) FAILED: %s\n',err.message); ok=false; end
%% (2) learn_hedge_units
try
    rng(2); [dYg,frg,Ug]=df.sim.learn_hedge_units(cfg,N,'geometric',5);
    rng(2); [dYs,frs,Us]=df.sim.learn_hedge_units(cfg,N,'sale',NaN);
    c3=abs(sum(dYg)-1)<1e-9 && abs(sum(dYs)-1)<1e-9 && sum(Ug.L)==N && sum(Us.L)==N && Ug.J_eff<=Ug.J+1e-9 && Us.J_eff<=Us.J+1e-9 ...
        && all(Ug.L>0) && all(Us.L>0) && isfinite(Ug.L1) && isfinite(Us.L1) && abs(sum(Ug.w)-1)<1e-9 && all(isfinite(frg(:))) && all(isfinite(frs(:)));
    fprintf('(2) learn_hedge_units: ok=%d; geometric(5): J=%d J_eff=%.0f meanL=%.2f L1=%.4f; sale: J=%d J_eff=%.0f meanL=%.2f L1=%.4f\n', ...
        c3,Ug.J,Ug.J_eff,Ug.mean_L,Ug.L1,Us.J,Us.J_eff,Us.mean_L,Us.L1);
    disp('   sale: mean duration by (player 1 type, player 2 type):'); disp(round(Us.mean_L_by_pair,2));
    ok=ok&&c3;
catch err, fprintf('(2) FAILED: %s\n',err.message); ok=false; end
%% (3) eps_player_cap: the learner-only region contains the joint region
try
    o=base; o.precomputed_distY={dYc}; rng(3); rj=df.stages.run_stage_ii(cfg,o);
    o.eps_player_cap=2; rng(3); ro=df.stages.run_stage_ii(cfg,o);
    nj=nnz(rj.VV_all(1,:)<=IDTOL); no=nnz(ro.VV_all(1,:)<=IDTOL);
    c4=all(isfinite(rj.VV_all(:))) && all(isfinite(ro.VV_all(:))) && no>=nj && all(ro.VV_all(1,:)<=rj.VV_all(1,:)+1e-7);
    fprintf('(3) eps_player_cap: joint %d cells, own %d cells (own contains joint: %d); truth feasible joint %d, own %d\n', ...
        nj,no,c4,rj.VV_all(1,1)<=IDTOL,ro.VV_all(1,1)<=IDTOL);
    ok=ok&&c4;
catch err, fprintf('(3) FAILED: %s\n',err.message); ok=false; end
%% (4) cons_N_override: a smaller composition count enlarges the region
try
    o=base; o.precomputed_distY={dYg}; rng(4); rN=df.stages.run_stage_ii(cfg,o);
    o.cons_N_override=Ug.J_eff; rng(4); rJ=df.stages.run_stage_ii(cfg,o);
    nN=nnz(rN.VV_all(1,:)<=IDTOL); nJ=nnz(rJ.VV_all(1,:)<=IDTOL);
    c5=all(isfinite(rJ.VV_all(:))) && nJ>=nN && all(rJ.VV_all(1,:)<=rN.VV_all(1,:)+1e-7);
    fprintf('(4) cons_N_override: at N %d cells, at J_eff=%.0f %d cells (monotone: %d)\n',nN,Ug.J_eff,nJ,c5);
    ok=ok&&c5;
catch err, fprintf('(4) FAILED: %s\n',err.message); ok=false; end
%% (5) first stage and v05_solve_eta on tiny grids
try
    lab=struct('actions_vec',actions_vec,'mu_true',mu,'sigma2_true',sigma2,'s_val',s,'IDTOL',IDTOL);
    rng(5); k=randsample((1:size(cfg.A,1))',2000,true,dYg(:,1)); P=cfg.A(k,:); eta0=alpha;
    e1=exp(eta0*P(:,1)); e2=exp(eta0*P(:,2)); D=1+e1+e2; u=rand(2000,1);
    y=zeros(2000,1); y(u<e1./D)=1; y(u>=e1./D & u<(e1+e2)./D)=2;
    nll=@(eta) -sum((y==1).*(eta*P(:,1))+(y==2).*(eta*P(:,2))-log(1+exp(eta*P(:,1))+exp(eta*P(:,2))));
    eta_hat=fminbnd(nll,-3,0); h=1e-4; se=1/sqrt((nll(eta_hat+h)-2*nll(eta_hat)+nll(eta_hat-h))/h^2);
    c6=abs(eta_hat-eta0)<0.3 && isfinite(se) && se>0;
    [VVw,dpw]=v05_solve_eta(-0.5,[1;linspace(0.5,2.5,3)'],[1;linspace(0.15,3.5,3)'],N,0.045,0.005,14,dYg,lab);
    [VVl,dpl]=v05_solve_eta(eta0,[1;linspace(0.9,1.1,3)'],[1;linspace(0.9,1.1,3)'],N,0.035,0.005,14,dYg,lab);
    c7=all(isfinite(VVw)) && all(isfinite(VVl)) && size(dpw,1)==numel(VVw) && size(dpl,1)==numel(VVl);
    fprintf('(5) first stage: eta_hat=%.4f se=%.4f (truth %.4f) ok=%d; v05_solve_eta: wide %d cells (%d feasible), local %d cells (%d feasible), finite=%d\n', ...
        eta_hat,se,eta0,c6,numel(VVw),nnz(VVw<=IDTOL),numel(VVl),nnz(VVl<=IDTOL),c7);
    ok=ok&&c6&&c7;
catch err, fprintf('(5) FAILED: %s\n',err.message); ok=false; end
if ok, fprintf('SMOKE_V05B: PASS\n'); else, fprintf('SMOKE_V05B: FAIL\n'); exit(1); end
