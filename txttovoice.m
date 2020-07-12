clear;
clc;
fid=fopen('train.txt','rt');
for t=1:37
textsent=fgetl(fid);    
y=tts(textsent,[],-1,16000);
fs=16000;
sound(y,fs);
str = input('Enter Filename   :','s');% accepting filename string from user
audiowrite(strcat('Training_Data\',str),y,16000);
end
fclose(fid);