
function VP = find_polytope_switch(maxiters,conf2,switch_eps)

global alpha AA s Egrid Psi marg_distrib NAct NPlayers
paths = df_repo_paths();

%% Grid of directions
NG = 100;
pp = haltonset(4,'Skip',1e3,'Leap',1e2);
PG0 = net(pp,NG);
k = 2;
norms = zeros(NG,1);
PG = [PG0(:,1).*k-k/2, PG0(:,2).*k-k/2, PG0(:,3).*k-k/2, PG0(:,4).*k-k/2];

for i = 1 : NG
    norms(i) = norm(PG(i,:));
end    

PGrid = PG ./ repmat(norms,1,4);

% for each direction in grid, send to AMPL the data of the problem, and call the AMPL .run file to
% generate polytope edges

eps = epsilon_switch(maxiters,conf2,switch_eps);
d4 = repmat(marg_distrib,1,NAct*NPlayers).*repmat(sqrt(marg_distrib),1,NAct*NPlayers).*repmat(eps',1,NAct*NPlayers);
% epsil=0.000000001;
epsil=1;

for i = 1 : NG
        
            lambda = PGrid(i,:);
            
            % PRINT .DAT FILE 
            fid = fopen(fullfile(paths.src, 'Polytope_Pricing.dat'), 'wt');    % this generates the data EXCLUDING THETA
            
            fprintf (fid, '# Data generated: %s\n\n', datestr(now));  
            fprintf (fid, 'data; \n\n'); 
            % 1 Game parameters
            fprintAmplParam(fid, 'alpha',alpha, 1);
            fprintAmplParam(fid, 'A1H',AA(2), 1);
            fprintAmplParam(fid, 'A2H',AA(2), 1);
            fprintAmplParam(fid, 'A1L',AA(1), 1);
            fprintAmplParam(fid, 'A2L',AA(1), 1);

            % 2 Epsilon Shocks
            fprintAmplParam(fid, 'nR',s, 1);

            % 3 direction of optimization
            fprintAmplParam(fid, 'lambda',lambda, 1);

            % 6 Payoff Shocks Grid
            fprintAmplParam(fid, 'Egrid',Egrid, 1);

            % 7 Prob Mass of Payoff Shocks
            fprintAmplParam(fid, 'Mass',Psi, 1);
            
            % 8 Epsilon
            fprintAmplParam(fid, 'epsilon',epsil, 1);
            fprintAmplParam(fid, 'd',d4, 1);

            fclose(fid);
            
            % call AMPL
            strAmplCommand = paths.ampl_command;
            outname = fullfile(paths.legacy_ampl_output, 'Polytope_Pricing.out');
            strAmplSystemCall = sprintf('"%s" "%s" > "%s"', strAmplCommand, fullfile(paths.src, 'Polytope_Pricing_new.run'), outname);
            [status,result] = system(strAmplSystemCall);
            
            % Display Progress
            i
            
end

%% Equalities and Inequalities BCE

PHH = csvread(fullfile(paths.legacy_ampl_output, 'PHH.sol'));
PHL = csvread(fullfile(paths.legacy_ampl_output, 'PHL.sol'));
PLH = csvread(fullfile(paths.legacy_ampl_output, 'PLH.sol'));
PLL = csvread(fullfile(paths.legacy_ampl_output, 'PLL.sol'));

VP = [PHH,PHL,PLH,PLL];

% Delete these files?

delete(fullfile(paths.legacy_ampl_output, 'PHH.sol'))
delete(fullfile(paths.legacy_ampl_output, 'PHL.sol'))
delete(fullfile(paths.legacy_ampl_output, 'PLH.sol'))
delete(fullfile(paths.legacy_ampl_output, 'PLL.sol'))

end
