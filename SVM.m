clear all, close all

N=1000;
p = [0.35,0.65]; % class priors for labels 0 and 1 respectively
label = rand(1,N) >= p(1); l = 2*(label-0.5);
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class

K=10;

x = zeros(2,N);
mu = zeros(2,1); Sigma = eye(2);
x(:,label==0) = randGaussian(Nc(1),mu,Sigma);%class -1 samples

theta = zeros(1,N);
radius = zeros(1,N);
theta(label==1) = -pi+2*pi.*rand(1,Nc(2));
radius(label==1) = 2+rand(1,Nc(2));
x(:,label==1) = [radius(label==1).*cos(theta(label==1)); radius(label==1).*sin(theta(label==1))];%class +1 samples

figure(1)
plot(x(1,label==0),x(2,label==0),'b.'), hold on
plot(x(1,label==1),x(2,label==1),'r.'), axis equal
xlabel('x1'), ylabel('x2')
legend('class: -1','class: 1')
title('Samples')


dummy = ceil(linspace(0,N,K+1));%Linear-SVM
for k = 1:K
    indPartitionLimits(k,1) = dummy(k)+1;
    indPartitionLimits(k,2) = dummy(k+1);
end
CList = 10.^linspace(-3,7,11);
for CCounter = 1:length(CList)
    C = CList(CCounter);
    for k = 1:K
        indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];%test set label 
        xValidate = x(:,indValidate); % Using folk k as validation set(test set )
        lValidate = l(indValidate);%the label of test set
        if k == 1
            indTrain = [101:N];
        elseif k == K
            indTrain = [1:900];
        else
            indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):1000];
        end% choose the training set
        xTrain = x(:,indTrain); lTrain = l(indTrain);%the label of train set
        SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','linear');
        dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
        indCORRECT = find(lValidate.*dValidate == 1); %the original label is same as the calculated label 
        inderror = find(lValidate.*dValidate == -1);%the original label is different from the calculated label 
        Ncorrect(k)=length(indCORRECT);
        Nerror(k)=length(inderror);
    end 
    PCorrect(CCounter)= sum(Ncorrect)/N; 
    Perror(CCounter) = sum(Nerror)/N;
end 
figure(2), subplot(1,2,1),
plot(log10(CList),Perror,'.',log10(CList),Perror,'-'),
xlabel('log_{10} C'),ylabel('K-fold Validation Error Estimate'),
title('Linear-SVM Cross-Val Error Estimate'),
[min_error1,indi] = min(Perror);
CBest1= CList(indi); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest1,'KernelFunction','linear');
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
indincorrect = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
trainingerror1=length(indincorrect);
Pro_trainingError1 = length(indincorrect)/N;
figure(2), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,indincorrect),x(2,indincorrect),'r.'), axis equal,
title('Training Data'),
xlabel('x1'), ylabel('x2'), legend('correct','incorrect');

dummy = ceil(linspace(0,N,K+1));% Gaussian-SVM
for k = 1:K, indPartitionLimits(k,:) = [dummy(k)+1,dummy(k+1)]; end
CList = 10.^linspace(-1,9,11); sigmaList = 10.^linspace(-2,3,13);
for sigmaCounter = 1:length(sigmaList)
    sigma = sigmaList(sigmaCounter);
    for CCounter = 1:length(CList)
        C = CList(CCounter);
        for k = 1:K
            indValidate = [indPartitionLimits(k,1):indPartitionLimits(k,2)];
            xValidate = x(:,indValidate); % Using folk k as validation set
            lValidate = l(indValidate);%the label of test set
            if k == 1
                indTrain = [101:N];
            elseif k == K
                indTrain = [1:900];
            else
                indTrain = [1:indPartitionLimits(k-1,2),indPartitionLimits(k+1,1):1000];
            end
            %choose the training set
            xTrain = x(:,indTrain); lTrain = l(indTrain);
            SVMk = fitcsvm(xTrain',lTrain,'BoxConstraint',C,'KernelFunction','gaussian','KernelScale',sigma);
            dValidate = SVMk.predict(xValidate')'; % Labels of validation data using the trained SVM
            indincorrect = find(lValidate.*dValidate == -1); 
            Nincorrect(k)=length(indincorrect);
        end 
        Perror(CCounter,sigmaCounter)= sum(Nincorrect)/N;
    end 
end
figure(3), subplot(1,2,1),
contour(log10(CList),log10(sigmaList),Perror',20); xlabel('log_{10} C'), ylabel('log_{10} sigma'),
title('Gaussian-SVM Cross-Val Error Estimate'), axis equal,
[min_error2,indi] = min(Perror(:)); [indBestC, indBestSigma] = ind2sub(size(Perror),indi);
CBest2= CList(indBestC); sigmaBest= sigmaList(indBestSigma); 
SVMBest = fitcsvm(x',l','BoxConstraint',CBest2,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
inderror = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
trainingerror2=length(inderror);
Pro_trainingError2 = length(inderror)/N;
figure(3), subplot(1,2,2), 
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,inderror),x(2,inderror),'r.'), axis equal,
title('Training Data '),
xlabel('x1'), ylabel('x2'),legend('correct','incorrect');

label = rand(1,N) >= p(1); l = 2*(label-0.5);%genernate the samples from the same distribution before
Nc = [length(find(label==0)),length(find(label==1))]; % number of samples from each class
x = zeros(2,N);
x(:,label==0) = randGaussian(Nc(1),mu,Sigma);
theta(label==1) = -pi+2*pi.*rand(1,Nc(2));
radius(label==1) = 2+rand(1,Nc(2));
x(:,label==1) = [radius(label==1).*cos(theta(label==1)); radius(label==1).*sin(theta(label==1))];

SVMBest = fitcsvm(x',l','BoxConstraint',CBest1,'KernelFunction','linear');
d = SVMBest.predict(x')'; % apply Linear-SVM with best hyperparameter to the new samples
inderror = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
Pro_trainingError3 = length(inderror)/N; %the probability of error estimate
figure(4),
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,inderror),x(2,inderror),'r.'), axis equal,
xlabel('x1'), ylabel('x2'), legend('correct','incorrect'),
title('Testing samples: Linear-SVM');


SVMBest = fitcsvm(x',l','BoxConstraint',CBest2,'KernelFunction','gaussian','KernelScale',sigmaBest);
d = SVMBest.predict(x')'; % Labels of training data using the trained SVM
inderror = find(l.*d == -1); % Find training samples that are incorrectly classified by the trained SVM
indCORRECT = find(l.*d == 1); % Find training samples that are correctly classified by the trained SVM
Pro_trainingError4 = length(inderror)/N; % the probability of error estimate
figure(5),
plot(x(1,indCORRECT),x(2,indCORRECT),'g.'), hold on,
plot(x(1,inderror),x(2,inderror),'r.'), axis equal,
xlabel('x1'), ylabel('x2'),legend('correct','incorrect'),
title('Testing samples: Gaussian-SVM'),


function x = randGaussian(N,mu,Sigma)
% Generates N samples from a Gaussian pdf with mean mu covariance Sigma
n = length(mu);
z =  randn(n,N);
A = Sigma^(1/2);
x = A*z + repmat(mu,1,N);
end