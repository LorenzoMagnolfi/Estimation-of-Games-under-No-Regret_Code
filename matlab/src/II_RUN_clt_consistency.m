%% II_RUN_clt_consistency  (Niccolo Proposal 4: sharper consistency set)
%  Compare the conservative L1 (Bretagnolle-Huber-Carol) consistency set against the
%  CLT chi-square ellipsoid, at s=5 and s=20, full-feedback RM, N=4M. The L1 radius
%  carries a ln(2^d) term (d = s^2) that blows up at large support; the CLT ellipsoid
%  uses the actual multinomial covariance and should be far tighter at s=20, where L1
%  ballooned the region to ~50%. Learns one RM trajectory per s and re-solves under
%  each consistency set. Incremental save per s. NO figures.
%  NOTE: the CLT set is ASYMPTOTIC (chi-square) -- a sharper but not finite-sample
%  bound; reported alongside the conservative L1, per the note's "report both".
clear; clc;
paths = df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2);
IDTOL=1e-8; aR=0.045; aC=0.005;
s_list=[5 20]; ngrid_list=[60 30];
rows=[]; res_store=struct();
out_path=fullfile(paths.artifacts,'clt_consistency.mat');

for si=1:numel(s_list)
    s=s_list(si); NGRID=ngrid_list(si);
    cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
    fgM=[1;linspace(0.18,1.5,NGRID)']; fgV=[1;linspace(0.15,3.5,NGRID)'];
    base=struct('maxiters_values',4000000,'alpha_set',0.05,'alpha_R',aR,'alpha_C',aC, ...
        'switch_eps',10,'backend','fast','adaptive',false,'require_corrected',true, ...
        'learning_style','rm','fixed_gridparamV',fgV,'fixed_gridparamM',fgM);
    fprintf('\n========== s=%d: L1 then CLT consistency ==========\n', s);
    o=base; o.consistency_slack_kind='L1'; rng(12345); res_l1=df.stages.run_stage_ii(cfg,o); traj=res_l1.distY_time_all;
    o=base; o.consistency_slack_kind='CLT'; o.precomputed_distY=traj; rng(12345); res_clt=df.stages.run_stage_ii(cfg,o);
    [a_l1,sh_l1,hb_l1]=region_area(res_l1,1,IDTOL);
    [a_clt,sh_clt,hb_clt]=region_area(res_clt,1,IDTOL);
    fprintf('s=%d: L1 area=%.4f share=%.4f hb=%d | CLT area=%.4f share=%.4f hb=%d\n', ...
        s, a_l1,sh_l1,hb_l1, a_clt,sh_clt,hb_clt);
    rows=[rows; struct('s',s,'L1_area',a_l1,'L1_share',sh_l1,'L1_hb',hb_l1, ...
        'CLT_area',a_clt,'CLT_share',sh_clt,'CLT_hb',hb_clt)]; %#ok<AGROW>
    res_store.(sprintf('s%d_L1',s))=res_l1; res_store.(sprintf('s%d_CLT',s))=res_clt;
    save(out_path,'rows','res_store','IDTOL','aR','aC','-v7.3');   % incremental per s
end

summary=struct2table(rows);
fprintf('\n=== Prop 4: CLT vs L1 consistency, s=5 and s=20 (N=4M) ===\n'); disp(summary);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'clt_consistency.csv'));
fprintf('Artifact: %s\n========== CLT vs L1 complete ==========\n', out_path);

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
