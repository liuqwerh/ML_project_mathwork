clear all;clc;
delta = 1; % tolerance for EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
N=10000; % Generate 10000 samples from a 4-component GMM
alpha_true = [0.1,0.2,0.3,0.4];
mu_true = [10 -10 -10 10;-10 -10 10 10];
Sigma_true(:,:,1) = [2,1;1,1.5];
Sigma_true(:,:,2) = [1.5,1;1,1];
Sigma_true(:,:,3) = [2.5,1;1,1.5];
Sigma_true(:,:,4) = [1.5,1;1,2];%set four different mu and sigma
x = randGMM(N,alpha_true,mu_true,Sigma_true);

logLikelihood2sum=0;

for i=1:10
    N=10000;
    if i==1
        b=x(:,(N/10)*i+1:N);
        y=b;
        y2=x(:,((i-1)*(N/10)+1):i*(N/10));
    elseif i==10
        a=x(:,1:(N/10)*(i-1));
        y=a;
        y2=x(:,((i-1)*(N/10)+1):i*(N/10));
    else
        a=x(:,1:(N/10)*(i-1));
        b=x(:,(N/10)*i+1:N);
        y=[a,b];
        y2=x(:,((i-1)*(N/10)+1):i*(N/10));
    end
     %y is training set and y2 is test set
N=N*0.9;
[d,M] = size(mu_true); % determine dimensionality of samples and number of GMM components
M=1; %create 6 components
alpha = ones(1,M)/M;
shuffledIndices = randperm(N);
mu = y(:,shuffledIndices(1:M)); % pick M random samples as initial mean estimates
[~,assignedCentroidLabels] = min(pdist2(mu',y'),[],1); % assign each sample to the nearest mean
for m = 1:M % use sample covariances of initial assignments as initial covariance estimates
    Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels==m))') + regWeight*eye(d,d);
end
t = 0; %displayProgress(t,x,alpha,mu,Sigma);

Converged = 0; % Not converged at the beginning
while ~Converged
    for l = 1:M
        temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(y,mu(:,l),Sigma(:,:,l));
    end
    plgivenx = temp./sum(temp,1);
    alphaNew = mean(plgivenx,2);
    w = plgivenx./repmat(sum(plgivenx,2),1,N);
    muNew = y*w';
    for l = 1:M
        v = y-repmat(muNew(:,l),1,N);
        u = repmat(w(l,:),d,1).*v;
        SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
    end
    Dalpha = sum(abs(alphaNew-alpha'));
    Dmu = sum(sum(abs(muNew-mu)));
    DSigma = sum(sum(abs(abs(SigmaNew-Sigma))));
    Converged = ((Dalpha+Dmu+DSigma)<delta); % Check if converged
    alpha = alphaNew; mu = muNew; Sigma = SigmaNew;
    t = t+1; 
    displayProgress(t,y,alpha,mu,Sigma);
    summ=0;
    for k=1:(N/0.9)/10
        h=alphaNew(1,:)*evalGaussian(y2(:,k),muNew(:,1),SigmaNew(:,:,1));
        
        summ=summ+log(h);
    end
end
end
logLikelihood2 = summ/10;

function displayProgress(t,x,alpha,mu,Sigma)
figure(2),
if size(x,1)==2
    subplot(1,2,1), cla,
    plot(x(1,:),x(2,:),'b.'); 
    xlabel('x_1'), ylabel('x_2'), title('Data and Estimated GMM Contours'),
    axis equal, hold on;
    rangex1 = [min(x(1,:)),max(x(1,:))];
    rangex2 = [min(x(2,:)),max(x(2,:))];
    [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2);
    contour(x1Grid,x2Grid,zGMM); axis equal, 
    subplot(1,2,2), 
end

logLikelihood = sum(log(evalGMM(x,alpha,mu,Sigma)));
plot(t,logLikelihood,'b.'); hold on,
xlabel('Iteration Index'), ylabel('Log-Likelihood of Data'),
drawnow; pause(0.1),
end


function x = randGMM(N,alpha,mu,Sigma)
d = size(mu,1); % dimensionality of samples
cum_alpha = [0,cumsum(alpha)];
u = rand(1,N); x = zeros(d,N); labels = zeros(1,N);
for m = 1:length(alpha)
    ind = find(cum_alpha(m)<u & u<=cum_alpha(m+1)); 
    x(:,ind) = randGaussian(length(ind),mu(:,m),Sigma(:,:,m));
end
end

function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end

function [x1Grid,x2Grid,zGMM] = contourGMM(alpha,mu,Sigma,rangex1,rangex2)
x1Grid = linspace(floor(rangex1(1)),ceil(rangex1(2)),101);
x2Grid = linspace(floor(rangex2(1)),ceil(rangex2(2)),91);
[h,v] = meshgrid(x1Grid,x2Grid);
GMM = evalGMM([h(:)';v(:)'],alpha, mu, Sigma);
zGMM = reshape(GMM,91,101);
end


%%%
function gmm = evalGMM(x,alpha,mu,Sigma)
gmm = zeros(1,size(x,2));
for m = 1:length(alpha) % evaluate the GMM on the grid
    gmm = gmm + alpha(m)*evalGaussian(x,mu(:,m),Sigma(:,:,m));
end
end

%%%
function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end