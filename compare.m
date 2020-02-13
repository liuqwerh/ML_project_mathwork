clear all, close all,

n = 2; % number of feature dimensions
N = 999; % number of iid samples
mu(:,1) = [-3;0]; mu(:,2) = [3;0];
Sigma(:,:,1) = [3 1;1 20]; Sigma(:,:,2) = [7 1;1 2];
p = [0.3,0.7]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1);
%decision=zeros(1,N);
decision2=zeros(1,N);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(n,N); 
for l = 0:1
    x(:,label==l) = mvnrnd(mu(:,l+1),Sigma(:,:,l+1),Nc(l+1))';
end
figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), axis equal,
legend('Class 0','Class 1'), 
title('Data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 


lambda = [0 1;1 0];
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); %threshold
discriminantScore = log(evalGaussian(x,mu(:,2),Sigma(:,:,2)))-log(evalGaussian(x,mu(:,1),Sigma(:,:,1)));% - log(gamma);
decision = (discriminantScore >= log(gamma));
ind00 = find(decision==0 & label==0); p00 = length(ind00)/Nc(1); % probability of true negative
ind10 = find(decision==1 & label==0); p10 = length(ind10)/Nc(1); % probability of false positive
ind01 = find(decision==0 & label==1); p01 = length(ind01)/Nc(2); % probability of false negative
ind11 = find(decision==1 & label==1); p11 = length(ind11)/Nc(2); % probability of true positive
error_count3 = [p10,p01]*Nc'; %error_number of MAP classifier

figure(2),
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,legend('right 0','mistake 10','mistake 01','right1'),title('map');


Sb = (mu(:,1)-mu(:,2))*(mu(:,1)-mu(:,2))';
Sw = Sigma(:,:,1) + Sigma(:,:,2);
[V,D] = eig(inv(Sw)*Sb); 
[~,ind] = sort(diag(D),'descend');
wLDA = V(:,ind(1)); % Fisher LDA projection vector
yLDA = wLDA'*x; % All data projected on to the line spanned by wLDA
wLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*wLDA; % ensures class1 falls on the + side of the axis
yLDA = sign(mean(yLDA(find(label==1)))-mean(yLDA(find(label==0))))*yLDA; % flip yLDA accordingly
range=wLDA'*mu;
bLDA=range(1):0.0001:range(2);
P=zeros(size(bLDA));
for i=1:size(bLDA,2)
    resLDA=yLDA+bLDA(i);    
    decision1 = (resLDA >= 0);
    ind10 = find(decision1==1 & label==0); 
    lda10 = length(ind10);
    ind01 = find(decision1==0 & label==1); 
    lda01 = length(ind01);
    P(i)=(lda10+lda01)/999;
end
[value,bLDA_final]=min(P);
bLDA_final=bLDA_final*0.0001+range(1);
u=wLDA'*x+bLDA_final;
for i=1:999
    if u(1,i)<0
        decision(1,i)=0;
    else
        decision(1,i)=1;
    end
end
error_count=0;
for i=1:999
    if decision(1,i)~=label(1,i)
        error_count=error_count+1;
    end
end% calculate the error_number of LDA classifier
figure(3), clf,
plot(u(find(label==0)),zeros(1,Nc(1)),'o'), hold on,
plot(u(find(label==1)),zeros(1,Nc(2)),'+'), axis equal,
plot(0,zeros(1,N),'dk');%Mark the 0 point on the plot, in black color
legend('Class 0','Class 1','classifier'), 
title('LDA projection of data and their true labels'),
xlabel('x_1'), ylabel('x_2'), 
ind00 = find(decision==0 & label==0);
ind10 = find(decision==1 & label==0);
ind01 = find(decision==0 & label==1);
ind11 = find(decision==1 & label==1);
figure(5),clf,
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,legend('right 0','mistake 10','mistake 01','right1'),title('LDA classifier');




b=x(:,find(label==0));
c=x(:,find(label==1));
y_x=sum(log((1-(1+exp(wLDA'*b+bLDA_final))).^(-1)));
y_xx=sum(log(1+exp(wLDA'*c+bLDA_final).^(-1)));
loglikelihood=y_x+y_xx;
fun = @(t)-sum(log((1+exp(t(1:2,:)'*x(:,find(label==1))+t(3,:))).^(-1)))-sum(log(1-(1+exp(t(1:2,:)'*x(:,find(label==0))+t(3,:))).^(-1)));
%options = optimset('CheckGradients',wLDA,'CheckGradients',bLDA_final);

x0 =[wLDA;bLDA_final];
z = fminsearch(fun,x0);
error_count2=0;

for i=1:999
    y3=(1+exp(z(1:2,:)'*x(:,i)+z(3,:))).^(-1);
    if y3>0.5
        decision2(1,i)=1;
    else
        decision2(1,i)=0;
    end
end
for i=1:999
    if decision2(1,i)~=label(1,i)
        error_count2=error_count2+1;
    end
end

ind00 = find(decision2==0 & label==0); 
ind10 = find(decision2==1 & label==0); 
ind01 = find(decision2==0 & label==1); 
ind11 = find(decision2==1 & label==1); 

figure(4), 
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'or'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
axis equal,legend('right 0','mistake 10','mistake 01','right1'),title('Logistic regression function');

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
