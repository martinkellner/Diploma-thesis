% spocita RF shifty a spravi histogramy
function RFshifts()
    
    figure;
    w = 480;
    h = 360;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w h]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w h]);
    set(gca, 'FontSize',18);
   
    hidden = loadHidden();    
    [activations] = loadActivations();
        
    % pomocne pre gain fieldy
    idx = 1:25:9*25;    
    x = repmat(-1:1:1, 1, 3);
    y = [ones(1,3) ones(1,3)-1 ones(1,3)-2];                  
    
    
        
    for h=1:hidden
                                                                                
        for i=0:24                  
          a = activations(idx+i,h);                       
          rfx(i+1) = (x*a)/sum(a);
          rfy(i+1) = (y*a)/sum(a);                              
        end                                       
         scatter(rfx, rfy, 1000, 'ro','LineWidth', 15);      
         axis([-1 1 -1 1])  
         
         hshift(h) = abs(max(rfx)-min(rfx));
         hstd(h) = std(rfx);
         xlabel(sprintf('shift = %0.2f, mean = %0.2f, std = %0.2f', hshift(h), mean(rfx), hstd(h) ));
         
         vshift(h) = abs(max(rfy)-min(rfy));
         vstd(h) = std(rfy);
         ylabel(sprintf('shift = %0.2f, mean = %0.2f, std = %0.2f', vshift(h), mean(rfy), vstd(h) ));
                 
         outname = sprintf('rf-shifts/%03d.png',h);  
         print(outname,'-dpng');                  
         
    end;
    
    centers = 0:0.1:1;
    ax = [-0.05 1.05 0 45];
    hist(hshift,centers);
    axis(ax);
    print('rf-shifts/hshift','-dpng');
    
    hist(hstd,centers);
    axis(ax);
    print('rf-shifts/hstd','-dpng');
    
    hist(vshift,centers);
    axis(ax);
    print('rf-shifts/vshift','-dpng');
    
    hist(vstd,centers);
    axis(ax);
    print('rf-shifts/vstd','-dpng');
    
    shift = [hshift, vshift];
    hist(shift,centers);
    axis(ax);
    title('absolute shift')
    print('rf-shifts/shift','-dpng');
    
    
    astd = [hstd, vstd];
    hist(astd,centers);
    axis(ax);
    title('std')
    print('rf-shifts/std','-dpng');
    
    
    save('rf-shifts/h-shift', 'hshift', '-ASCII');
    save('rf-shifts/v-shift', 'vshift', '-ASCII');        
    save('rf-shifts/h-std', 'hstd', '-ASCII');
    save('rf-shifts/v-std', 'vstd', '-ASCII');            
                    
end