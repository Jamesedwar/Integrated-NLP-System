function p=hmm_probob(H,ob)
%Compute Probability of Observations.
T=size(ob,2);
[Qi,Aij,Bit]=H.getprob(ob);
alpha(:,1)=Qi.*Bit(:,1);
P(1)      =1/sum(alpha(:,1));
alpha(:,1)=alpha(:,1)*P(1);
for k=1:T-1
    alpha(:,k+1)=(transpose(Aij)*alpha(:,k)).*Bit(:,k+1);
    P(k+1)=1/sum(alpha(:,k+1));
    alpha(:,k+1)=alpha(:,k+1)*P(k+1);
end

p=-sum(log(P));
