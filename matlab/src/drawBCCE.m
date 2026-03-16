function plot = drawBCCE(file,VP)

COORD = VP(:,[2 3 1]);
Pred = Polyhedron('V', COORD);

Simpl = [1,0,0;0,1,0;0,0,1;0,0,0];
SimplP = Polyhedron('V',Simpl);

figure('Name','Equilibrium Prediction','NumberTitle','off','GraphicsSmoothing','on');

h1 = Pred.plot('Alpha',0.3,'Color',[1 0 0],'LineJoin','chamfer');
hold on

h2 = SimplP.plot('Alpha',0.1,'Color','green','LineJoin','chamfer');

xl=xlabel('$q\big(p_{\ell},p_h\big)$');
set(xl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)

yl=ylabel('$q\big(p_h,p_{\ell} \big)$') ;
set(yl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)

zl = zlabel('$q\big(p_h,p_h \big)$');
set(zl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',24)


xticks([0 0.5 1 ])
yticks([0 0.5 1 ])
set(gca, 'TickLabelInterpreter', 'latex', 'FontName', 'cmr10', 'FontSize', 21, 'Box', 'off');


view(105,35)

% Save Coordinate Information

saveas(gcf,file,'epsc');

end
