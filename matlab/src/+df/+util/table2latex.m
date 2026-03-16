function latex_str = table2latex(T, caption, label)
    % Start the table
    latex_str = ['\begin{widetable}{.98\columnwidth}{l' repmat('c', 1, width(T)) '} '];
    latex_str = [latex_str '\toprule '];
    
    % Add header
    header = ['& ' strjoin(strrep(T.Properties.VariableNames, '_', ' '), ' & ')];
    latex_str = [latex_str header ' \\ '];
    latex_str = [latex_str '\midrule '];
    
    % Add data
    for i = 1:height(T)
        row = strrep(T.Properties.RowNames{i}, '_', ' ');
        for j = 1:width(T)
            if iscell(T{i,j})
                cellContent = T{i,j}{1};
                if isempty(cellContent) || (isnumeric(cellContent) && isnan(cellContent)) || strcmp(cellContent, '\cdot')
                    row = [row ' & $\cdot$'];
                elseif isnumeric(cellContent)
                    row = [row ' & $' sprintf('%.1f', cellContent) '$'];
                elseif ischar(cellContent) || isstring(cellContent)
                    % Check if the content is an interval
                    if contains(cellContent, '[') && contains(cellContent, ']')
                        row = [row ' & $' strrep(cellContent, '_', ' ') '$'];
                    else
                        row = [row ' & ' strrep(cellContent, '_', ' ')];
                    end
                else
                    row = [row ' & ' strrep(num2str(cellContent), '_', ' ')];
                end
            elseif isnumeric(T{i,j})
                if isnan(T{i,j})
                    row = [row ' & $\cdot$'];
                else
                    row = [row ' & $' sprintf('%.1f', T{i,j}) '$'];
                end
            else
                row = [row ' & ' strrep(num2str(T{i,j}), '_', ' ')];
            end
        end
        latex_str = [latex_str row ' \\ '];
    end
    
    % End the table
    latex_str = [latex_str '\bottomrule '];
    latex_str = [latex_str '\end{widetable} '];
end