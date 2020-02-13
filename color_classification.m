clear all, close all

delta = 1e-5; % tolerance for K-Means and EM stopping criterion
regWeight = 1e-10; % regularization parameter for covariance estimates
Pic = imread('2222.jpg');%input the plane image
Picc = double(Pic)/255;%preprocessing the picture
[row,col,lay] = size(Picc);%read the width,length and layers(numbers of color) of the image
N = row*col; % number of samples
piccc = zeros(row,col,lay);
x = zeros(5,N); 
color = zeros(3,N);

for j = 1:col
    for i = 1:row
        x(1,i+(j-1)*row) = (j-1)/col; % x coordinate
        x(2,i+(j-1)*row) = (i-1)/row; % y coordinate
        x(3,i+(j-1)*row) = Picc(i,j,1); % R
        x(4,i+(j-1)*row) = Picc(i,j,2); % G
        x(5,i+(j-1)*row) = Picc(i,j,3); % B
    end
end
%normalize 5 feature vectors
%store 5 feature vectors of the image into x matrix

for Seg = 2:5    %K-Means
    shuffledIndices = randperm(N);
    meann = x(:,shuffledIndices(1:Seg)); % pick K random samples as initial mean estimates
    meannNew = meann;
    [~,assignedCentroidLabels] = min(pdist2(meann',x'),[],1); % assign each sample to the nearest mean
    Converged = 0;
    while ~Converged
        [~,assignedCentroidLabels] = min(pdist2(meann',x'),[],1); % assign each sample to the nearest mean again
        for m = 1:Seg
            meannNew(:,m)= mean(x(:,find(assignedCentroidLabels == m)),2);
        end
        dmean = sum(sum(abs(meannNew-meann)));
        Converged = (dmean < delta); % Check if converged
        meann = meannNew;  
    end
    for m = 1:Seg
        color(1,find(assignedCentroidLabels == m)) = 0.1*m; % assign different color to different segments
        color(2,find(assignedCentroidLabels == m)) = 0.15*m;
        color(3,find(assignedCentroidLabels == m)) = 0.2*m; 
    end
    for z = 1:3
        for j= 1:col
            for i = 1:row
                piccc(i,j,z) = color(z,i+(j-1)*row);
            end
        end
    end    
    I3=im2uint8(piccc);
    figure(Seg-1),
    image(I3),
    title(['K-Means ','   ',' segments:',num2str(Seg)])
end,

[d,M] = size(meann);% GMM-model EM algorithm
for Seg = 2:5
    Mapset = zeros(Seg,N);
    alpha = ones(1,Seg)/Seg;
    shuffledIndices = randperm(N);
    meann = x(:,shuffledIndices(1:Seg)); % pick K random samples as initial mean estimates
    [~,assignedCentroidLabels] = min(pdist2(meann',x'),[],1); % assign each sample to the nearest mean
    for m = 1:Seg % use sample covariances of initial assignments as initial covariance estimates
        Sigma(:,:,m) = cov(x(:,find(assignedCentroidLabels == m))') + regWeight*eye(d,d);
    end
    Converged = 0;
    while ~Converged
        for l = 1:Seg
            temp(l,:) = repmat(alpha(l),1,N).*evalGaussian(x,meann(:,l),Sigma(:,:,l));
        end
        plgivenx = temp./sum(temp,1);
        alphaNew = mean(plgivenx,2);
        w = plgivenx./repmat(sum(plgivenx,2),1,N);
        meannNew = x*w';
        for l = 1:Seg
            v = x-repmat(meannNew(:,l),1,N);
            u = repmat(w(l,:),d,1).*v;
            SigmaNew(:,:,l) = u*v' + regWeight*eye(d,d); % adding a small regularization term
        end
        Dalpha = sum(abs(alphaNew-alpha));
        dmean = sum(sum(abs(meannNew-meann)));
        DSigma = sum(sum(sum(abs(SigmaNew-Sigma))));
        Converged = ((Dalpha+dmean+DSigma)<delta); % Check if converged
        alpha = alphaNew; meann = meannNew; Sigma = SigmaNew;
    end
    for m = 1:Seg %Map-classification
        Mapset(m,:) = log(evalGaussian(x,meann(:,m),Sigma(:,:,m))*alpha(m));
    end
    [Max,dec] = max(Mapset);
    for m = 1:Seg
        color(1,find(dec == m)) = 0.1*m; % assign different color to different segments
        color(2,find(dec== m)) = 0.15*m;
        color(3,find(dec == m)) = 0.2*m;
        %color(:,find(D == k)) = 0.1*k;%repmat(mu(3:5,k),1,length(find(D == k))); % assign mu value to classified x
    end
    for j = 1:col
        for i = 1:row
            for m = 1:3
                piccc(i,j,m) = color(m,i+(j-1)*row);
            end
        end
    end
    I4=im2uint8(piccc);
    figure(Seg+3),
    image(I4),
    title(['EM   ' 'COMPONENTS:' num2str(Seg)])
end

function g = evalGaussian(x,mu,Sigma)
% Evaluates the Gaussian pdf N(mu,Sigma) at each coumn of X
[n,N] = size(x);
invSigma = inv(Sigma);
C = (2*pi)^(-n/2) * det(invSigma)^(1/2);
E = -0.5*sum((x-repmat(mu,1,N)).*(invSigma*(x-repmat(mu,1,N))),1);
g = C*exp(E);
end
