% Vykresli do grafu vahy vsetkych vystupnych neuronov na skryte neurony.
function plotOuts()
    
    outputs = 19;         % 19 pre X a 19 pre Y
    
    w = 320;
    h = 240;
    set(gcf, 'PaperUnits', 'points');
    set(gcf, 'PaperSize', [w*2 h*outputs]);
    set(gcf, 'PaperPositionMode', 'manual');
    set(gcf, 'PaperPosition', [0 0 w*2 h*outputs]);
    
    out = loadOutputWeights(outputs);
                
    for i=0:(2*outputs-1)
                                        
        pos = mod(i,outputs)*2+1 + floor(i/outputs);                        
        subplot(outputs,2,pos)
        
        plotWeights(out(:,i+1), 5);
        if i>=outputs
            xlabel('y')
        else
            xlabel('x')
            ylabel(num2str(mod(i,outputs)))
        end
                
    end
        
    print('out/out.png','-dpng');


end