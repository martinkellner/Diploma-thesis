% Vykresli aktivacie skrytych neuronov: 
%  - pozicia grafu oznacuje poziciu visualneho stimulu
%  - v ramci jedneho grafu su aktivacie pre rozne gaze
% Parametre su nepovinne:
%  - indexes defautne 1:hidden
%  - outputName defaultne 'vis-fixed-visual/fixedvisual_%03d.png'
function gmFixedVisual(indexes, outputName)
         
    w = 480;
    h = 360;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*5 h*3]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*5 h*3]);
    
    hidden = loadHidden();
    hsteepness = 0.05;
    output = 19;
    [activations, choosen, tilts, versions, factorGaze, factorVisual, hbiases] = loadActivations();
    out = loadOutputWeights(output);    
    [visualLeft, visualRight] = loadVisualInput(choosen);
    
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
    vmax = 60;
    tmin = -35;
    tmax = 15;   
    
    if (nargin==0)
        indexes = 1:hidden;        
    end;
    
    for ih=1:length(indexes)
        h = indexes(ih);
        
        [lhid, rhid, thid, vhid] = loadHiddenWeights(h-1);
        
        %colormap('default');
        
        subplot(3,5,1)
        plotReceptiveField(lhid);         
        title('left receptive field')        
        %freezeColors
        
        subplot(3,5,2)
        plotReceptiveField(rhid);         
        title('right receptive field')        
        %freezeColors
        
        subplot(3,5,6)
        plotWeights(thid, 1, [0.5 0 0]);         
        title('weights of connections to TILT neurons (vertical angle)')        
                
        subplot(3,5,7)
        plotWeights(vhid, 1, [0 0.5 0]);                                                            
        title('weights of connections to VERSION neurons (horizontal angle)')        
        
        subplot(3,5,11)
        plotWeights(out(h,output+1:output+output), 5, [0.5 0.5 0.5]);
        title('weights of connections to output neurons coding vertical angle (Y)')        
        
        subplot(3,5,12)
        plotWeights(out(h,1:output), 5, [0.5 0.5 0.5]);
        title('weights of connections to output neurons coding horizontal angle (X)')                
              
        %colormap(winter)
        
%         actGaze = zeros(rows, 1);
%         for t=1:tn
%             for v=1:vn
%                 ii = (t-1)*vn+v; 
%                 sumGaze = (tilts(t,2:end)*thid + versions(v,2:end)*vhid)*factorGaze * hsteepness;                
%                 actGaze(ii) = factor/(1+exp(-2*sumGaze));                
%             end
%         end                
        
        for d=1:p
            
            j = floor((d-1)/3)*2 + 2 + d;
                     
            subplot(3,5,j)
            
            scatter([xv; 60], [yt; 10], [f factor], 'o', 'MarkerEdgeColor', [0 0.5 0]);
            
            
            axis([vmin vmax tmin tmax]);            
            set(gca, 'ycolor', [0.5 0 0]);
            set(gca, 'xcolor', [0 0.5 0]);
            set(gca,'YTick',tilts(:,1));
            set(gca,'XTick',versions(:,1));
            hold on                        
            
            s = (activations(1+(d-1)*rows:d*rows,h).*factor);      
            
            sumVis = (visualLeft(d,:)*lhid + visualRight(d,:)*rhid)*factorVisual;            
            sumVis = (sumVis + hbiases(d)) * hsteepness;
            actVis = factor/(1+exp(-2*sumVis));                                    
            
%            plus  = zeros(rows,1)+1e-6;
%            minus = zeros(rows,1)+1e-6;    

%            plus( (s-actVis)>0 ) = actVis;
%            minus( (s-actVis)<0 ) = actVis;
%             for i=1:rows                
%                 diff = (s(i)-actVis(i));                 
%                 if diff>0
%                     plus(i) = actVis(i);
%                 else 
%                     minus(i) = actVis(i);                    
%                 end                
%             end                                                                                                                          
            
            scatter(xv, yt, s, [0 0 0.6], 'filled');

            %scatter(xv, yt, actGaze, 'o', 'MarkerEdgeColor', [0.7 0.7 0.7]);
            
            idx = (s-actVis)>0;
            scatter(xv(idx), yt(idx), f(idx)/4, '+', 'w', 'LineWidth', 2);
            %idx = (s-actVis)<0;
            %scatter(xv(idx), yt(idx), f(idx)/2, 'x', 'w', 'LineWidth', 1);
            scatter(xv, yt, actVis, 'o', 'MarkerEdgeColor', 'm');
            scatter(60, 10, actVis, 'o', 'filled', 'MarkerFaceColor', 'm');
            
            hold off

        end;         
               
        %ulozim obrazok
        if (nargin>=2)
            outname = outputName;
        else               
            outname = sprintf('vis-fixed-visual/fixedvisual_%03d.png',h);      
        end        
        print(outname,'-dpng');                  
        
    end; 
    

end

%             for t=1:tn
%                 for v=1:vn                                
%                     ii = (t-1)*vn+v;
%                     sumGaze = (tilts(t,2:end)*thid + versions(v,2:end)*vhid)*factorGaze; 
%                     koef = abs(s(ii)/(sumVis+sumGaze)) / 2;
%                     if sumVis > 0
%                         sVisPos(ii) = sumVis*koef;
%                     else
%                         sVisNeg(ii) = -sumVis*koef;
%                     end
%                     if sumGaze > 0
%                         sGazePos(ii) = sumGaze*koef;
%                     else
%                         sGazeNeg(ii) = -sumGaze*koef;
%                     end                    
%                 end
%             end   

            %scatter(xv, yt, ups, '^', 'MarkerEdgeColor', [1 1 0]);
            %scatter(xv, yt, downs,'v','MarkerEdgeColor', [1 1 0]);

           
           % scatter(xv, yt, actVis,  cv, 'LineWidth', 2);
            %scatter(xv, yt, actVisPos, '^',  'r', 'LineWidth', 2);
            %scatter(xv, yt, actVisNeg, 'v',  'r', 'LineWidth', 2);
            
%             scatter(xv, yt, sVisPos, 's', 'r', 'LineWidth',1.5);
%             scatter(xv, yt, sVisNeg, 'd', 'r', 'LineWidth',1.5);                                   
%             scatter(xv, yt, sGazePos, '+', 'b', 'LineWidth',1.5);
%             scatter(xv, yt, sGazeNeg, 'x', 'b', 'LineWidth',1.5);            
            
%             if d==p
%                 legend('maximum possible activity','real activity',...
%                     'positive visual stimulus','negative visual stimulus',...
%                     'positive gaze stimulus','negative gaze stimulus',...
%                     'Location', 'EastOutside');
%             end