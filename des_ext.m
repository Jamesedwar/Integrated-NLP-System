x=wavread('WAV\Praveen-01.wav');
z = Preprocessing(x);
wlen=120;
Fs=160000;
Q=12;
%Feature extraction
   [m,n]=size(z);
  disp(m);
  disp(n);
   N=2*2^nextpow2(wlen);
%    disp('N');
%    disp(N);
   [m1,n1]=size(N);
%    disp(m1);
%   disp(n1);
w=hamming(wlen);
[m2,n2]=size(w);
% disp('w');
%    disp(m2);
%   disp(n2);
X=buffer(z,wlen,wlen*2/3);
[m3,n3]=size(X);
%    disp(m3);
%   disp(n3);
%   disp('Buffer');
% disp(X);
 disp('Length');
  L=size(X,2);
disp(L)
H=melfb(Q,N,Fs);
disp('H');
disp(H);
[m4,n4]=size(H);
disp(m4);
disp(n4);
for k=1:L
  y=fft(w.*X(:,k),N);
  disp(y);
  s=H*abs(y(1:N/2+1)).^2;
  disp('s');
  disp(s)
  Y(:,k)=dct(log(s));
end
   
   disp(Y)
   [m5,n5]=size(Y);
   disp(m5);
   disp(n5);
   for col=1:n5
   for row=1:m5
    E(col)=sum(abs(Y(row,col)).^2);
   end
   end
   disp(E);
        
%     x = 2*len ;
%     E(m) = sum(abs(x).^2);%Energy of a signal
        
    
%    C=mfcc(z,fs,12,120); %12mfcc +Energy coefficient
%    CC=[C;E'];
%    D = mfcc2delta(CC,2); %13 Delta coefficients
%    DD = mfcc2delta(D,2); %13 Double Delta coefficients
%    F=[CC;D;DD]; 