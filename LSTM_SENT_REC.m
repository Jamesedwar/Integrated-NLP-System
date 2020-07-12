clear;
clc;
fid1=fopen('sample3.txt');
tline1 = fgetl(fid1);
tlines1 = cell(0,1);
double s1;
s1={' ',' '};
G1=cell(11,1);
A1=cell(11,1);
H= {'Good Morning Praveen';
'This is Samsung S6';
'The device has touch screen';
'There are 2 cameras';
'It has Android Operating System';
'It supports up to 64 GB';
'It supports Bluetooth';
'It has long-range WiFi';
'It supports Office and Pdf';
'The WiFi is Naturally OFF';
'It has Video and Photo editing'
};

while ischar(tline1)
    tlines1{end+1,1} = tline1;
    tline1 = fgetl(fid1);
end
n=length(tlines1);
 
for i=1:n
       t=tlines1{i};
       s1 = strtrim(t);
       for index=1:length(s1)
       s1(s1==' ') = '';
       end
       s=log(sum(s1));
       disp(s);
       G1{i}=s;
end

numObservations = numel(G1);
for i=1:numObservations
    sequence = A1{i};
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths,idx] = sort(sequenceLengths);

H = H(idx);
% disp(G);
% disp(H);  

%TEST DATA
fid2=fopen('sample3.txt');
tline2 = fgetl(fid2);
tlines2 = cell(0,1);
double s2;
s2={' ',' '};
G2=cell(11,1);
A2=cell(11,1);
while ischar(tline2)
    tlines2{end+1,1} = tline2;
    tline2 = fgetl(fid2);
end
n=length(tlines2);
 
for i=1:n
       t=tlines2{i};
       s2 = strtrim(t);
       for index=1:length(s2)
       s2(s2==' ') = '';
       end
       s=log(sum(s2));
       disp(s);
       G{i}=s;
end

numObservations = numel(G2);
for i=1:numObservations
    sequence = A2{i};
    sequenceLengths(i) = size(sequence,2);
end
[sequenceLengths,idx] = sort(sequenceLengths);
disp('IDX');
disp(idx);
H = H(idx);
 
inputSize = 1;
outputSize = 12;
outputMode = 'last';
numClasses = 11;
layers = [ ...
    sequenceInputLayer(inputSize)
    lstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
maxEpochs =100;
miniBatchSize = 10;
shuffle = 'never';
options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);
%X=DATASET;

H=categorical(H);
disp(H);
disp('Training Started');
tic;
net = trainNetwork(G,H,layers,options);
toc;
disp('Training Complete');
YPred=classify(net,G,'MiniBatchSize',27);
disp(YPred);

 

 