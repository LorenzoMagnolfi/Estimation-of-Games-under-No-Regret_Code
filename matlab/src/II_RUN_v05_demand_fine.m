%% II_RUN_v05_demand_fine  As II_RUN_v05_demand with a 31 x 31 cost grid (closes the resolution hole of the 16 x 16 run).
%  For each candidate demand slope eta on a 39-point grid, rebuild Pi(eta), the BCCE constraints, the per-candidate
%  tailF tolerances (family F at delta = alpha_R/M_R with the latent-count ceiling on the L1 event; K(eta) enters
%  through the payoff span), and the L1 consistency radius; then test all (mu, sigma^2) cost candidates on a 16x16
%  grid. One Hedge trajectory at N = 4M generated under the true slope eta_0 = -1/3 (seed 12345), as in July's
%  II_RUN_demand_L1, which used the legacy radius (no M_R) and regret matching. Also run the first-moment rectangle
%  (code 16) for the valid-Markov comparison. Direct SeDuMi lane with parfor. Incremental save per eta; CSV; no figures.
%  This is the known-payoff joint-identification exercise; the C3 projection with a first-stage demand set is a
%  separate design (plan item 5).
clear; clc; paths=df_repo_paths();
eta_true=-1/3; eta_grid=linspace(-1.0,-0.05,39); actions_vec=[4;5;6;7;8]; mu_true=[3;3]; sigma2_true=eye(2); s_val=5;
T=4000000; alpha_R=0.045; alpha_C=0.005; IDTOL=1e-8;
NGridM=30; NGridV=30;   % finer cost grid: 961 cells per slope
gridparamM=[1;linspace(0.55,mu_true(1)*0.5,NGridM)']; gridparamV=[1;linspace(0.15,sigma2_true(1,1)*3.5,NGridV)'];
cfg_true=df.setup.game_simulation(2,eta_true,actions_vec,mu_true,sigma2_true,s_val);
rng(12345); t0=tic; [dY,~,fr]=df.sim.learn_hedge(cfg_true,1,T,T,1,1,1,1); fprintf('[DGP] Hedge learned %.0fs, max realized regret %.3e\n',toc(t0),max(fr(:)));
action_distribution=dY;
[distpars,distribution_parameters]=df.report.build_param_grid(mu_true,sigma2_true,gridparamM,gridparamV,'Both');
NGrid=size(distpars,1);
ncpu=str2double(getenv('SLURM_CPUS_ON_NODE'));
if isfinite(ncpu) && ncpu>1 && exist('gcp','file')==2 && isempty(gcp('nocreate')), try, parpool('Processes',min(ncpu,16)); catch, end, end
codes=[14 16]; code_names={'tailF','momentF'};
n_eta=numel(eta_grid);
VV_by_eta=zeros(numel(codes),n_eta,NGrid); n_identified=zeros(numel(codes),n_eta); truth_in=false(numel(codes),n_eta); K_by_eta=zeros(n_eta,s_val);
out=fullfile(paths.artifacts,'v05_demand_fine.mat');
for ei=1:n_eta
    eta_cand=eta_grid(ei);
    cfg_test=df.setup.game_simulation(2,eta_cand,actions_vec,mu_true,sigma2_true,s_val);
    cstr=df.solvers.build_constraints(cfg_test.type_space,cfg_test.action_space,cfg_test.Pi);
    dim_u=cstr.NA-1; a_dim=cstr.a; NAg=cstr.NAg; s2=cstr.s2; T_sorted=cstr.T_sorted;
    cons_idx=dim_u+cstr.NA+(1:s2);
    r_L1=df.solvers.compute_consistency_slack(s2,alpha_C,T,'L1');
    K_by_eta(ei,:)=max(cfg_test.Pi(:,:,1))-min(cfg_test.Pi(:,:,1));
    Psi=zeros(s2,NGrid); marg=zeros(s_val,NGrid);
    for nd=1:NGrid
        Psi(:,nd)=mvnpdf(T_sorted,distribution_parameters{2,nd},distribution_parameters{3,nd});
        mv=distribution_parameters{2,nd}; mv=mv(1); sv=distribution_parameters{3,nd}; sv=sv(1,1);
        md=normpdf(cfg_test.type_space{1,1},mv,sqrt(sv)); marg(:,nd)=md/sum(md);
    end
    Psi=Psi./sum(Psi,1);
    for ci=1:numel(codes)
        c_all=zeros(size(cstr.B_EQ,2),NGrid);
        for nd=1:NGrid
            eps_t=df.solvers.compute_epsilon_v05(cfg_test,T,marg(:,nd)',r_L1,codes(ci),alpha_R);
            eps_fin=repmat(eps_t,1,NAg*a_dim);
            c_all(:,nd)=[zeros(1,dim_u),action_distribution(:,1)',Psi(:,nd)',1,eps_fin]';
        end
        gopts=struct('verbose',false,'solver','sedumi','cons_l1_r',r_L1,'cons_l1_idx',cons_idx,'use_parfor',true);
        [VV,~,~]=df.solvers.solve_grid_sedumi(cstr,c_all,gopts);
        VV_by_eta(ci,ei,:)=VV(:)'; n_identified(ci,ei)=nnz(VV<=IDTOL); truth_in(ci,ei)=VV(1)<=IDTOL;
        fprintf('[eta %2d/%d] eta=%.3f %-8s: %d/%d identified (truth cell %d)\n',ei,n_eta,eta_cand,code_names{ci},n_identified(ci,ei),NGrid,truth_in(ci,ei));
    end
    save(out,'eta_grid','eta_true','VV_by_eta','n_identified','truth_in','distpars','K_by_eta','T','s_val','alpha_R','alpha_C','codes','code_names','IDTOL','NGrid','fr','-v7.3');
end
rows=table();
for ci=1:numel(codes)
    rows=[rows; table(repmat(string(code_names{ci}),n_eta,1),eta_grid(:),n_identified(ci,:)',truth_in(ci,:)','VariableNames',{'code','eta','n_identified','truth_cell_identified'})]; %#ok<AGROW>
    id=eta_grid(n_identified(ci,:)>0);
    if ~isempty(id), fprintf('%s: identified eta range [%.4f, %.4f] (true %.4f), %d of %d eta values\n',code_names{ci},min(id),max(id),eta_true,numel(id),n_eta); end
end
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(rows,fullfile(paths.tables_ii,'v05_demand_fine.csv'));
fprintf('Artifact: %s\n========== V05 demand (fine grid) complete ==========\n',out);
