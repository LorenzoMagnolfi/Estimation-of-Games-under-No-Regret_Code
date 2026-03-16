function plots = drawRegrets_per_period(filename,regret_per_period)

global s tps

figure('Name','Regrets','NumberTitle','off','GraphicsSmoothing','on');

hold on;

endp = 2000;

startp  = 200 ;

y_L_Pl1 = regret_per_period(:,floor(s/3),1);
y_H_Pl1 = regret_per_period(:,floor(s*2/3),1);

y_L_Pl2 = regret_per_period(:,floor(s/4),2);
y_H_Pl2 = regret_per_period(:,floor(s*3/4),2);

y_L_Pl1 = y_L_Pl1(startp:endp);
y_H_Pl1 = y_H_Pl1(startp:endp);

y_L_Pl2 = y_L_Pl2(startp:endp);
y_H_Pl2 = y_H_Pl2(startp:endp);

x_L_Pl1 = 1:length(y_L_Pl1);
x_H_Pl1 = 1:length(y_H_Pl1);

x_L_Pl2 = 1:length(y_L_Pl2);
x_H_Pl2 = 1:length(y_H_Pl2);

P3 = plot(x_H_Pl2,y_H_Pl2,'*','Color', [0.2 0.8 0],'linewidth',2);
P4 = plot(x_L_Pl2,y_L_Pl2,'*','Color', [ 0 0 0.9 ],'linewidth',2);

P1 = plot(x_H_Pl1,y_H_Pl1,'--','Color', [0.4 0.4 0.4],'linewidth',2);
P2 = plot(x_L_Pl1,y_L_Pl1,'--','Color',[0.7 0.3 0.1] ,'linewidth',2);


% Settings
set(gcf,'color','white');
% title(['True ' plot_param ' = ' num2str(true_param) ', \epsilon Discretization = ' num2str(true_disc)]);
%cbar_labels = num2cell(iters);
%cbar = colorbar('Location','Eastoutside',...
%        'Ticks',iter_ticks,'TickLabels',cbar_labels);
%cbar.Label.String = 'Number of Iterations';
%colormap(plot_colors);


lgnd1l = sprintf('Seller 1, $t_1=%.2f$',tps(floor(s/3),1));
lgnd1h = sprintf('Seller 1, $t_1=%.2f$',tps(floor(s*2/3),1));
lgnd2l = sprintf('Seller 2, $t_2=%.2f$',tps(floor(s/4),1));
lgnd2h = sprintf('Seller 2, $t_2=%.2f$',tps(floor(s*3/4),1));

lgnd = legend([ P2 P1 P4 P3 ],lgnd1l,lgnd1h,lgnd2l,lgnd2h,'Location','northeast');
set(lgnd,'Interpreter', 'latex', 'FontName','cmr10','FontSize',16)


xl=xlabel('Number of periods $N$');
set(xl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',16)

yl=ylabel('Regret $R_N(i,t_i)$') ;
set(yl,'Interpreter', 'latex', 'FontName','cmr10','FontSize',16)

set(gca,'TickLabelInterpreter', 'latex');
set(gca,...
            'Units','normalized',...
             'FontName','cmr10',...
       'FontUnits','points',...
        'FontWeight','normal',...
        'FontSize',15,...
        'Box','off');

set(gca,'XTick',1:400:length(y_L_Pl1))
set(gca,'XTickLabel',startp:400:endp)
    
saveas(gcf,filename,'epsc');

end
