% Natrenuje 1d-som na aktivaciach - aby ich usporiadala
function somActivations()
    hidden = loadHidden();       
    
    [activations, ~, tilts, versions, ~, ~, ~] = loadActivations();
        
    activations = activations';
    sm = som_make(activations, 'msize', [hidden 1], 'tracking', 10, 'training', [250 350]);            
    figure;
    save('som/sm', '-struct', 'sm')
    plotActivationsAsGlyphs(sm.codebook);
    print('som/sm.png', '-dpng');
    
    w = 480;
    h = 360;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*3 h*3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*3 h*3]);            
    
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
    
    
    for i=1:hidden
    
        activations = sm.codebook(i,:).*factor;                
                
        for d=1:p
                                             
            subplot(3,3,d)
            scatter(xv, yt, f, [1 0 0]);
            
            axis([vmin vmax tmin tmax ]);
            set(gca,'YTick',tilts(:,1));
            set(gca,'XTick',versions(:,1));            
            
            hold on

            s = activations(1+(d-1)*rows:d*rows);      
            scatter(xv, yt, s, 'filled');
            hold off        

        end; 
                        
        %ulozim obrazok
        outname = sprintf('som/som%03d.png',i);  
        print(outname,'-dpng');
    end;
        

   
    
end






