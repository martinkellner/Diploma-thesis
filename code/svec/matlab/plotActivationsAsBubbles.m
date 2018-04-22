% Vykresli aktivacie skrytych neuronov: 
%  - pozicia grafu oznacuje poziciu visualneho stimulu
%  - v ramci jedneho grafu su aktivacie pre rozne gaze

function plotActivationsAsBubbles(activations)
              
    [~, ~, tilts, versions, ~, ~, ~] = loadActivations();    
    
    %pomocne pre gain fieldy    
    tn = size(tilts);
    tn = tn(1);
    vn = size(versions);
    vn = vn(1);
    rows = tn*vn;
    [yt, xv] = meshgrid(tilts(:,1), versions(:,1));   
    yt = yt(:);
    xv = xv(:);
    factor = 1000;
    f(1:rows) = factor;    
    p = length(activations)/rows;
    vmin = -50;  
    vmax = 50;
    tmin = -35;
    tmax = 15;   
            
        
    for d=1:p
                                           
        subplot(3,3,d)
            
        scatter(xv, yt, f, 'o', 'MarkerEdgeColor', [0 0.5 0]);
                        
        axis([vmin vmax tmin tmax]);            
        set(gca, 'ycolor', [0.5 0 0]);
        set(gca, 'xcolor', [0 0.5 0]);
        set(gca,'YTick',tilts(:,1));
        set(gca,'XTick',versions(:,1));
        hold on                        
            
        s = (activations(1+(d-1)*rows:d*rows).*factor);      
                               
        scatter(xv, yt, s, [0 0 0.6], 'filled');                                              
        hold off

    end;                                

end
