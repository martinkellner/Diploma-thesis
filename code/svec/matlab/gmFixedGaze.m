% nakresli gain modulation graf s fixovanym gaze angle a meniacou sa
% poziciou vizualneho stimulu
function gmFixedGaze()
    
    w = 240;
    h = 180;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*(2+5) h*5]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*(2+5) h*5]);
   
    hidden = loadHidden();    
    [activations, ~, tilts, versions] = loadActivations();
    factor = 250;
    
    % pomocne pre gain fieldy
    idx = 1:25:9*25;    
    x = repmat(1:3, 1, 3);
    y = [ones(1,3)+2 ones(1,3)+1 ones(1,3)];                  
    f(1:9) = factor;
    
    for h=1:hidden
        
        [lhid, rhid, thid, vhid] = loadHiddenWeights(h-1);
                        
        %subplot(5,7, [1 2 8 9])
        subplot(5,7, 1);
        plotReceptiveField(lhid)
        
        %subplot(5,7, [15 16 22 23])      
        subplot(5,7,2);
        plotReceptiveField(rhid)
        
        
        %subplot(5,7, 29)
        subplot(5,7, 8);
        plotWeights(thid);
        
        %subplot(5,7, 30)
        subplot(5,7, 9);
        plotWeights(vhid);
                        
        
        
        for i=0:24                  
          offset = 10-floor(i/5)*2; %2 prve stlpce
          index  = (5-floor(i/5))*5-4+mod(i,5); %pozicia, idem opacne riadkami, lebo tilt zacina -30          
          
          subplot(5,7,  offset + index);
          scatter(x, y, f, [0 0 0]);
          axis([0 4 0 4])
          set(gca,'YTick',[]);
          set(gca,'XTick',[]);
          xlabel(sprintf('version = %d', versions(mod(i,5)+1)));
          ylabel(sprintf('tilt = %d', tilts(floor(i/5)+1)));
          hold on;  
          a = activations(idx+i,h);
          scatter(x, y, a*factor, [0 0.3 1],'filled');
             
          rfx(i+1) = (x*a)/sum(a);
          rfy(i+1) = (y*a)/sum(a);
          scatter(rfx(i+1), rfy(i+1) , factor/4, 'filled', 'MarkerEdgeColor', [1 0 0], 'MarkerFaceColor', [1 0 0], 'LineWidth', 2);
          
          hold off;
        end       
        
        hold off;
        rfx = rfx-2;
        rfy = rfy-2;
        subplot(5,7, [22 23 29 30]);        
        scatter(rfx, rfy, factor/4, 'ro','LineWidth', 2);        
        axis([-1 1 -1 1])   
        %set(gca,'YTick',-2:1:2);
        %set(gca,'XTick',-2:1:2);
                
        xlabel(sprintf('shift = %0.2f, mean = %0.2f, std = %0.2f', abs(max(rfx)-min(rfx)), mean(rfx), std(rfx) ));
        ylabel(sprintf('shift = %0.2f, mean = %0.2f, std = %0.2f', abs(max(rfy)-min(rfy)), mean(rfy), std(rfy) ));
                
        outname = sprintf('vis-fixed-gaze/fixedgaze_%03d.png',h);  
        print(outname,'-dpng');                  
        
    end;
                    
end