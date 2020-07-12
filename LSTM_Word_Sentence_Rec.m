clear;
clc;
TRS={'Good.wav', 'Morning.wav', 'Praveen.wav', 'This.wav', 'is.wav', 'Samsung.wav', 'S6.wav', 'The.wav', 'Device.wav', 'has.wav', 'Touch.wav', 'Screen.wav', 'There.wav', 'are.wav', 'two.wav', 'Cameras.wav', 'It.wav', 'Android.wav', 'Operating.wav', 'System.wav', 'Supports.wav', 'up.wav', 'to.wav', 'Sixty_four.wav', 'GB.wav', 'Bluetooth.wav', 'Long_range.wav', 'WiFi.wav', 'Office.wav', 'and.wav', 'Pdf.wav', 'Naturally.wav', 'OFF.wav', 'video.wav', 'photo.wav', 'editing.wav'};
G1=cell(36,1);

W=zeros(1,36);
O=0;
H={'Good';
    'Morning';
    'Praveen';
    'This';
    'is';
    'Samsung';
    'S6';
    'The';
    'Device';
    'has';
    'Touch';
    'Screen';
    'There';
    'are';
    'two';
    'Cameras';
    'It';
    'Android';
    'Operating';
    'System';
    'Supports';
    'up';
    'to';
    'Sixty_four';
    'GB';
    'Bluetooth';
    'Long_range';
    'WiFi';
    'Office';
    'and';
    'Pdf';
    'Naturally';
    'OFF';
    'video';
    'photo';
    'editing'};

tic;
for k=1:length(TRS)
   
   
    
    [x,fs] = audioread(strcat('Training_Data\',TRS{k}));
    
 
    
% 2. Preprocessing
z = Preprocessing(x);
% 3. Feature Extraction - MFCC Coefficients
X = mfcc(z,fs,12,120);
C=X';


[L,B]=size(X);
G=zeros(1,B);
for p=1:L
for t=1:B
O=abs(O+X(p,t))/1.005;
end
end

G1(k)=num2cell(O);

end

% fileID = fopen('celldata.dat','w');
% [L,B]=size(C);
% for l=1:L
%     for m=1:B
% fprintf(fileID, '%f \t', C(l,m));
%     end
%     fprintf(fileID,'\n');
% end
% fclose(fileID);
disp(G1);
x=1:36;

H=H(x);
inputSize = 1;
nhu = 36;
outputMode = 'last';
numClasses = 36;
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(nhu,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
maxEpochs =20000;
miniBatchSize = 27;
shuffle = 'never';
options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);
%X=DATASET;

H=categorical(H);

disp('Training Started');
tic;
net = trainNetwork(G1,H,layers,options);

toc;
disp('Training Complete');

YPred=classify(net,G1,'MiniBatchSize',27);
disp(YPred);
  

 