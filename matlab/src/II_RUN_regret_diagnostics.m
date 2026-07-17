%% II_RUN_regret_diagnostics  Realized regret vs the envelope (D2 + memo §10.3).
%  Documents that the learners are themselves no-regret at the horizons we use:
%  final realized regret by horizon for the three learners (contextual RM,
%  proxy-regret/bandit PRM with the decaying floor, pooled RM), plus the
%  per-player regret paths (downsampled) for the envelope-vs-realized panel.
%  For the pooled learner both regret units are recorded: its own pooled unit
%  (should vanish) and the type-conditional unit (should not).
clear; clc;
paths=df_repo_paths();
NPlayers=2; alpha=-(1/3); actions_vec=[4;5;6;7;8]; mu=3*ones(2,1); sigma2=eye(2); s=5;
cfg=df.setup.game_simulation(NPlayers,alpha,actions_vec,mu,sigma2,s);
N_grid=[500000,1000000,2000000,4000000,8000000];
styles={'rm','prm','pooled'};
rows=cell(0,4); paths_ds=struct();
out=fullfile(paths.artifacts,'regret_diagnostics.mat');
for si=1:numel(styles)
    sty=styles{si};
    cfg.learning_style=sty;
    for ni=1:numel(N_grid)
        N=N_grid(ni);
        rng(12345+ni);
        t0=tic;
        if strcmp(sty,'prm')
            [~,~,fr,p1,p2]=learn_mod_prm(cfg,1,N,N,1,1,1,1);
            fr2=NaN;
        elseif strcmp(sty,'pooled')
            [~,~,fr,fr2,p1,p2]=learn_mod_pooled(cfg,1,N,N,1,1,1,1); %#ok<ASGLU>
        else
            [~,~,fr,p1,p2]=learn_mod(cfg,1,N,N,1,1,1,1);
            fr2=NaN;
        end
        t1=toc(t0);
        frs=max(fr(:));                       % worst component, conservative
        fr2s=max(fr2(:));
        rows(end+1,:)={sty,N,frs,fr2s}; %#ok<SAGROW>
        ds=max(1,floor(numel(p1)/2000));
        key=sprintf('%s_N%dk',sty,N/1000);
        paths_ds.(key)=struct('p1',p1(1:ds:end),'p2',p2(1:ds:end),'ds',ds);
        fprintf('%-6s N=%4dk: final_regret=%.3e  second_unit=%.3e  (%.1fs)\n', ...
            sty, N/1000, frs, fr2s, t1);
        save(out,'rows','paths_ds','-v7.3');   % incremental
    end
end
style=rows(:,1); N=cell2mat(rows(:,2));
final_regret=cell2mat(rows(:,3)); second_unit=cell2mat(rows(:,4));
summary=table(style,N,final_regret,second_unit);
if ~exist(paths.tables_ii,'dir'), mkdir(paths.tables_ii); end
writetable(summary, fullfile(paths.tables_ii,'regret_diagnostics.csv'));
fprintf('Artifact: %s\n========== regret diagnostics complete ==========\n', out);
