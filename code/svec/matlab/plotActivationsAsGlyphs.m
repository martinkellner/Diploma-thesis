function plotActivationsAsGlyphs(activations)
    s = size(activations);
    if (s(2)~=25*9)
        act = activations';
    else
        act = activations;
    end;
    
    w = 300;
    h = 300;
          
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*10 h*10]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*10 h*10]);
    
    s = size(act);
    tmp = ones(s(1), s(2));
    tmp2 = ones(s(1), s(2));
    tmp(:, 25:50) = 0.01;
    tmp2(:, 50:75) = 0.01;
    tmp(:, 75:100) = 0.01;
    tmp2(:,100:125) = 0.01;
    tmp(:,125:150) = 0.01;
    tmp2(:,150:175) = 0.01;
    tmp(:,175:200) = 0.01;
    tmp2(:,200:225) = 0.01;
            
      
    h = glyphplot(tmp, 'standardize','matrix', 'obslabels','');
    hold on;
    set(h, 'color', [0.8 1 0.8]);         
    
    h = glyphplot(tmp2, 'standardize','matrix', 'obslabels','');
    hold on;
    set(h, 'color', [1 0.8 0.8]);                 
    
    x = ones(s(1), s(2));
    x = x.*0.00001;
    x(1:s(1),25:25:s(2)) = 1;   
    h = glyphplot(x, 'standardize','matrix', 'obslabels','');    
    set(h, 'color', [1 0 0]);            
    
    glyphplot(act, 'standardize', 'matrix', 'LineWidth', 1.5);        
    
    hold off;
end