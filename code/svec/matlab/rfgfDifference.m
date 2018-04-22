% spocita rozdiely medzi orientaciou RF a GF, spravi histogram
function rfgfDifference()
         
    w = 480;
    h = 360;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w h]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w h]);
    
    hidden = loadHidden();
    hsteepness = 0.05;    
    [activations, choosen, tilts, versions, factorGaze, factorVisual, hbiases] = loadActivations();    
    [visualLeft, visualRight] = loadVisualInput(choosen);
    
     
    tn = size(tilts);
    tn = tn(1);
    vn = size(versions);
    vn = vn(1);
            
    rows = tn*vn;
    [yt, xv] = meshgrid(tilts(:,1), versions(:,1));   
    yt = yt(:);
    xv = xv(:);      
    p = length(activations)/rows;
    
    
    if (nargin==0)
        indexes = 1:hidden;        
    end;    
    
    for ih=1:length(indexes)
        h = indexes(ih);
        
        [lhid, rhid] = loadHiddenWeights(h-1);
        
        % pocitam RF uhol
        d = 1;       
        for y=3:-1:1
            for x=1:3
                s = (activations(1+(d-1)*rows:d*rows,h));                  
                sumVis = (visualLeft(d,:)*lhid + visualRight(d,:)*rhid)*factorVisual;            
                sumVis = (sumVis + hbiases(d)) * hsteepness;
                actVis = 1/(1+exp(-2*sumVis));                                 
                %ttt(d) = actVis;
                MRF(y,x) = actVis;
                d = d+1;
            end
        end;
        %MRF
        [rfy, rfx] = centerOfMass(MRF);
        rfy = rfy-2;
        rfx = rfx-2;
        
        %xx = repmat(1:3, 1,3);         
        %yy = [ones(1,3)+2 ones(1,3)+1 ones(1,3)]; 
        %rfx = (xx*ttt')/sum(ttt)-2
        %rfy = (yy*ttt')/sum(ttt)-2
        
        
        % pocitam GF uhol
        d = floor((1+p)/2);  % stredny obrazok                                                                      
        s = (activations(1+(d-1)*rows:d*rows,h));                          
        a = 1;
        for t=1:tn
            for v=1:vn
                MGF(t,v) = s(a);
                a = a+1;
            end;
        end;
        [gfy, gfx] = centerOfMass(MGF);
        gfx = gfx-3;
        gfy = gfy-3;
        
        % rozdiel                                
        CosTheta = dot([rfx,rfy],[gfx,gfy])/(norm([rfx,rfy])*norm([gfx,gfy]));
        ThetaInDegrees = acos(CosTheta)*180/pi;
        u(h) = ThetaInDegrees;
                       
    end; 
    
    hist(u);
    print('rfgf-difference/diff.png','-dpng');
    
    save('rfgf-difference/diffs', 'u', '-ASCII');          
    

end
