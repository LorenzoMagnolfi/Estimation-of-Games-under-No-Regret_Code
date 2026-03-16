function plot = drawConvergence(filename,numdists,M,N,actions,VP)

global A

figure('Name','Learning','NumberTitle','off','GraphicsSmoothing','on');

hold on;
Simpl = [1,0,0;0,1,0;0,0,1;0,0,0];
SimplP = Polyhedron('V',Simpl);
h2 = SimplP.plot('Alpha',0.1,'Color','green','LineJoin','chamfer');

hold off;
%% Plot Simplex

% Record Distributions
VP_2 = zeros(numdists,size(A,1));
ind1 = 1;

for obs = round(M * (1:numdists)/numdists )
    for ind2 = 1:size(A,1)
        VP_2(ind1,ind2) = sum(prod(actions(1:(N+obs),:) == A(ind2,:),2))/(N+obs); 
    end
    ind1 = ind1 + 1;
end

% Simplex
%figure('Name','Equilibrium Prediction','NumberTitle','off','GraphicsSmoothing','on'); hold on;
figure(1); hold on;
COORD = VP_2(:,[2 3 4]);

s = 10*ones(numdists,1);
c = linspace(0,1,numdists)';
scatter3(COORD(:,1),COORD(:,2),COORD(:,3),s,c);


%% draw Polyhedra of BCE Predictions and of Simplex

COORD = VP(:,[2 3 1]);
Pred = Polyhedron('V', COORD);

h1 = Pred.plot('Alpha',0.3,'Color',[1 0 0],'LineJoin','chamfer');


xl=xlabel('$q\big(p_{\ell},p_h\big)$');
set(xl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',16)

yl=ylabel('$q\big(p_h,p_{\ell} \big)$') ;
set(yl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',16)

zl = zlabel('$q\big(p_h,p_h \big)$');
set(zl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',16)


xticks([0 0.25 0.5 0.75 1 ])
yticks([0 0.25 0.5 0.75 1 ])
zticks([0 0.25 0.5 0.75 1 ])
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 15, 'Box', 'off');



%lgnd = legend([h2(1) h3],'$\cap_{y\in Y}BD\left(y,x\right)$','$y=\rho_{0}$','Location','northwest');
%set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',9)
view(105,35)

saveas(gcf,filename,'epsc');

% Remove Hold
hold off;

end