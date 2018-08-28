
[activations] = loadActivations();

%D = zeros(100, 2);
D = zeros(100, 5);
for ih=1:100
    data = reshape(activations(:,ih), 25, 9);
    
%     R = corrcoef(data);
%     D(ih, 1) = mean(mean(R));
%     D(ih, 2) = std(std(R));

    sdev = std(data);
    klas = (std(sdev) < 0.1) & (max(sdev) < 0.4);
    D(ih,:) = [ih mean(sdev) std(sdev) max(sdev) klas];
        
    

end
