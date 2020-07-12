clear ;
clc;
l=0;
TRS={'t1.wav', 't2.wav', 't3.wav', 't4.wav', 't5.wav', 't6.wav', 't7.wav', 't8.wav', 't9.wav', 't10.wav','t11.wav'};
TES={'s1.wav', 's2.wav', 's3.wav', 's4.wav', 's5.wav', 's6.wav', 's7.wav', 's8.wav', 's9.wav', 's10.wav','s11.wav'};

disp('Training Started');
tic;

for k=1:length(TRS)
    [x,fs] = audioread(strcat('Training_Data\',TRS{k}));
    
   
    % Feature extraction (feature vectors as columns)
     
% II-IV Preprocessing
   z = Preprocessing(x);
%    sound(z,fs);
%    pause(1);
% V - Feature Extraction - MFCC Coefficients
    
  X=mfcc(z,fs,12,120);
 %LSTM Layer
inputSize = 182;
outputSize = 11;
outputMode = 'last';
numClasses = 11;
layers = [ ...
    sequenceInputLayer(182)
    lstmLayer(outputSize,'OutputMode',outputMode)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer];
maxEpochs = 150;
miniBatchSize = 182;
shuffle = 'never';
options = trainingOptions('sgdm', ...
    'MaxEpochs',maxEpochs, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle', shuffle);
C= ['a';'b';'c';'d';'e';'f';'g';'h';'i';'j';'k';'l'];

R = cellstr(C);
disp(R);
%R=num2cell(C);
%  H=size(12,182);
 
 for t=1:12
 x=X(t,1:180);
 x=abs(ceil(x));
 H=num2cell(x(t));
%  Y = bsxfun(@eq, x(:), 1:max(x));
%  disp(Y);
%  end
%  H = rand(12,182);
% %Delete first and second column
% H(:,1) = [];
% H(:,2) = [];
% c1=ones(12,1);
%  c2=ones(12,1);
 
%Add new columns
% H = [c1 c2 x];
  
%    disp(H);
  % ch=num2str(X,2);
  % H=ch(1:12,1:180);
%   H=array2table(H,'RowNames',{'D1','D2','D3','D4','D5','D6','D7','D8','D9','D10','D11','D12'},'VariableNames',{'Predictors','Responses','v1','v2','v3','v4','v5','v6','v7','v8','v9','v10','v11','v12','v13','v14','v15','v16','v17','v18','v19','v20','v21','v22','v23','v24','v25','v26','v27','v28','v29','v30','v31','v32','v33','v34','v35','v36','v37','v38','v39','v40','v41','v42','v43','v44','v45','v46','v47','v48','v49','v50','v51','v52','v53','v54','v55','v56','v57','v58','v59','v60','v61','v62','v63','v64','v65','v66','v67','v68','v69','v70','v71','v72','v73','v74','v75','v76','v77','v78','v79','v80','v81','v82','v83','v84','v85','v86','v87','v88','v89','v90','v91','v92','v93','v94','v95','v96','v97','v98','v99','v100','v101','v102','v103','v104','v105','v106','v107','v108','v109','v110','v111','v112','v113','v114','v115','v116','v117','v118','v119','v120','v121','v122','v123','v124','v125','v126','v127','v128','v129','v130','v131','v132','v133','v134','v135','v136','v137','v138','v139','v140','v141','v142','v143','v144','v145','v146','v147','v148','v149','v150','v151','v152','v153','v154','v155','v156','v157','v158','v159','v160','v161','v162','v163','v164','v165','v166','v167','v168','v169','v170','v171','v172','v173','v174','v175','v176','v177','v178','v179','v180'});
%       disp(H);
 net = trainNetwork(H,R,layers,options);

end
end
%Testing Data