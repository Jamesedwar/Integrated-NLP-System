function H = hmm_reestimate(H,ob,iter)
% Reestimate parameters using EM method.
% H = hmm_reestimate(H,ob,iter)

% Set constants.
T = size(ob,2);

for k1 = 1:iter
   [Qi,Aij,Bit,Bitk] = H.getprob(ob);
   
   P(k1) = hmm_probob(H,ob);
   Pavg  = exp(-P(k1)/T);
   
   % Compute forward variables.
   alpha(:,1) = (Qi.*Bit(:,1))*Pavg;
   for k2 = 1:T-1
      alpha(:,k2+1) = (transpose(Aij)*alpha(:,k2)).*Bit(:,k2+1);
      alpha(:,k2+1) = alpha(:,k2+1)*Pavg;
   end
   
   % Compute backward variables.
   beta(:,T) = ones(H.Nst,1)*Pavg;
   for k2 = T:-1:2
      beta(:,k2-1) = Aij*(beta(:,k2).*Bit(:,k2));
      beta(:,k2-1) = beta(:,k2-1)*Pavg;
   end
   
   % Reestimate initial state probabilities.
   Qi = alpha(:,1).*beta(:,1);
   Qi = Qi/sum(Qi);
   
   % Reestimate state transition probabilities.
   Aij = Aij.*(alpha(:,1:T-1)*transpose(beta(:,2:T).*Bit(:,2:T)));
   Aij = bsxfun(@rdivide,Aij,sum(Aij,2));
   
   % Reestimate observation probabilities.
   kappa = bsxfun(@times,alpha.*beta,bsxfun(@rdivide,Bitk,Bit+realmin));
   for k2 = 1:H.Nst
      for k3 = 1:H.Nmix
         knorm = kappa(k2,:,k3)/(sum(kappa(k2,:,k3))+realmin);
         
         Bik(k2,k3).w = sum(kappa(k2,:,k3));
         Bik(k2,k3).m = sum(bsxfun(@times,knorm,ob),2);
         
         v = sum(bsxfun(@times,knorm,bsxfun(@minus,ob,H.Bik(k2,k3).m).^2),2);
         if norm(v) > 1e-3, Bik(k2,k3).C = diag(v); end
      end
      u             = cat(1,Bik(k2,:).w);
      u             = num2cell(u/sum(u));
      [Bik(k2,:).w] = deal(u{:});
   end
   
   H = HMM(Qi,Aij,Bik);
end

P(iter+1) = hmm_probob(H,ob);

%figure, plot(0:iter,P);
%title(sprintf('Initial log-probability: %.4g / Final log-probability: %.4g',P(1),P(iter+1)));
