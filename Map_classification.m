clear all;clc;

n = 2; % feature dimensions
N = 10000; % number of samples
u1 = [-1;0]; u2 = [1;0]; u3 = [0;1];
Sigma1 = [1 -0.4;-0.4 0.5]; Sigma2 = [0.5 0;0 0.2]; Sigma3 = [0.1 0;0 0.1];
p = [0.15,0.35,0.5]; % equal class priors
label = rand(1,N);% create label 0/1
for i=1:10000
    if label(1,i)<=0.15
    label(1,i)=0;
elseif (label(1,i)>0.15)&&(label(1,i)<=0.5)
    label(1,i)=1;
else
    label(1,i)=2;
    end;
end

Nc = [length(find(label==0)),length(find(label==1)),length(find(label==2))]; % number of samples from each class
x = zeros(n,N); 
a=Nc(1);
x(:,label==0)=mvnrnd(u1,Sigma1,Nc(1))';
x(:,label==1)=mvnrnd(u2,Sigma2,Nc(2))';
x(:,label==2)=mvnrnd(u3,Sigma3,Nc(3))';% Generate normal distributions

figure(1), clf,
plot(x(1,label==0),x(2,label==0),'o'), hold on,
plot(x(1,label==1),x(2,label==1),'+'), hold on,
plot(x(1,label==2),x(2,label==2),'*'),axis equal,
legend('Class 0','Class 1','Class 3 '), 
title('Gaussian distribution labels'),
xlabel('x_1'), ylabel('x_2'), 
lambda = [0 1;1 0];
decision= rand(1,N);
gamma = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(2); 
discriminantScore = log(evalGaussian(x,u2,Sigma2))-log(evalGaussian(x,u1,Sigma1));
gamma1= (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(3)/p(2); 
discriminantScore1 = log(evalGaussian(x,u2,Sigma2))-log(evalGaussian(x,u3,Sigma3));
j=0;
for i=1:10000
    if (discriminantScore(1,i)>=log(gamma))&&(discriminantScore1(1,i)>=log(gamma1))
        j=j+1;
        decision(1,i)=1;
    end
end
b=j;

gamma2 = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(2)/p(1); 
discriminantScore2 = log(evalGaussian(x,u1,Sigma1))-log(evalGaussian(x,u2,Sigma2));
gamma3= (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(3)/p(1); 
discriminantScore3 = log(evalGaussian(x,u1,Sigma1))-log(evalGaussian(x,u3,Sigma3));
j=0;
for i=1:10000
    if (discriminantScore2(1,i)>=log(gamma2))&&(discriminantScore3(1,i)>=log(gamma3))
        j=j+1;
        decision(1,i)=0;
    end
end


gamma4 = (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(2)/p(3); 
discriminantScore4 = log(evalGaussian(x,u3,Sigma3))-log(evalGaussian(x,u2,Sigma2));
gamma5= (lambda(2,1)-lambda(1,1))/(lambda(1,2)-lambda(2,2)) * p(1)/p(3); 
discriminantScore5 = log(evalGaussian(x,u3,Sigma3))-log(evalGaussian(x,u1,Sigma1));
j=0;
for i=1:10000
    if (discriminantScore4(1,i)>=log(gamma4))&&(discriminantScore5(1,i)>=log(gamma5))
        j=j+1;
        decision(1,i)=2;
    end
end

ind00 = find(decision==0 & label==0); 
ind10 = find(decision==1 & label==0);  
ind20 = find(decision==2 & label==0); 
ind01 = find(decision==0 & label==1); 
ind11 = find(decision==1 & label==1);  
ind21 = find(decision==2 & label==1); 
ind02 = find(decision==0 & label==2);  
ind12 = find(decision==1 & label==2); 
ind22 = find(decision==2 & label==2); 

confuma =[size(ind00,2),size(ind01,2),size(ind02,2);size(ind10,2),size(ind11,2),size(ind12,2);...
    size(ind20,2),size(ind21,2),size(ind22,2)];
mis_number= size(ind10,2)+ size(ind20,2)+ size(ind01,2)+ size(ind21,2)+ size(ind02,2)+ size(ind12,2);
mis_prob= mis_number/10000;

figure(2), % class 0 circle, class 1 +, correct green, incorrect red
plot(x(1,ind00),x(2,ind00),'og'); hold on,
plot(x(1,ind10),x(2,ind10),'+r'); hold on,
plot(x(1,ind20),x(2,ind20),'+r'); hold on,
plot(x(1,ind01),x(2,ind01),'+r'); hold on,
plot(x(1,ind11),x(2,ind11),'+g'); hold on,
plot(x(1,ind21),x(2,ind21),'+r'); hold on,
plot(x(1,ind02),x(2,ind02),'+r'); hold on,
plot(x(1,ind12),x(2,ind12),'+r'); hold on,
plot(x(1,ind22),x(2,ind22),'*g'); hold on,
axis equal,legend('right0','mistake10','mistake20','mistake01','right1','mistake21','mistake02','mistake12','right2'),title('Map Rule');



function g = evalGaussian(x,mu,Sigma)%define the evalGaussian function
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
C = ((2*pi)^n * det(Sigma))^(-1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(inv(Sigma)*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end


