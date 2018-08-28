% Nacita vahy z vystupnych na skryte neurony, vrati ich ako maticu, kde
% stlpec rerpezentuje vystupny neuron a riadok skryty.
function [ out ] = loadOutputWeights(outputs)
  
    if (nargin==0)        
        outputs = 19;         % 19 pre X a 19 pre Y
    end
                    
    for i=0:(2*outputs-1)
        
        filename = sprintf('out.%03d',i);                  
        o = load(filename);    
        
        out(:,i+1) = o(:,1);
    end        

end

