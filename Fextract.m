x=wavread('WAV\Praveen-01.wav');
z = Preprocessing(x);
wlen=120;
Fs=16000;
Q=12;
%Feature extraction
  
     C=mfcc1(z,Fs,12,120); %12mfcc +Energy coefficien
     [m5,n5]=size(C);
     disp(m5);
     disp(n5);
     D = mfcc2delta(C,2); %13 Delta coefficients
     DD = mfcc2delta(D,2); %13 Double Delta coefficients
     F=[C;D;DD]; 
     [m6,n6]=size(F);
     disp(m6);
     disp(n6);