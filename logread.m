%% generate data
K = 3;
N = 30;
D = 2;
centers = 10*randn(K,D);
shapes = randn(K,D,D);
points = zeros(N,D);
for n=1:N
    k = randi(K,1,1);
    points(n,:) = squeeze(centers(k,:))' + ...
        squeeze(shapes(k,:,:))*randn(D,1);
end
%%
clf;
plot(points(:,1),points(:,2),'.');
%% write to file
K = 3;
N = 30;
D = 2;
fout = fopen('data.in','w');
fprintf(fout,'%d %d %d\n',N,D,K);
fprintf(fout,'%f\n',points');
%% run program
!./a.out
%% read in file
!python fixlog.py
fin = fopen('fixedlog.txt','r');
while ~feof(fid)
    status = fscanf(fin,'%d',1);
    if isempty(status)
        break
    end
    fprintf(1,'status: %d\n',status);
    if status == 0 || status == 1
        N=fscanf(fin,'%d',1);
        D=fscanf(fin,'%d',1);
        K=fscanf(fin,'%d',1);
        fprintf(1,'params: %d %d %d\n',N,D,K);
        mu = zeros(K,D);
        sigma = zeros(K,D,D);
        for k=1:K
            mu(k,:)=fscanf(fin,'%f',D);
            sigma(k,:,:)=fscanf(fin,'%f',[D D]);
            disp('mu:');
            disp(squeeze(mu(k,:)));
            disp('sigma:');
            disp(squeeze(sigma(k,:,:)));
        end
        clf; hold on;
        plot(data(:,1),data(:,2),'.');
        for k=1:K
            plot(mu(k,1),mu(k,2),'r.');
            drawgaussian(mu(k,1:2),sigma(k,1:2,1:2),'g');
        end
        
    end
    if status == 1
        assignments = fscanf(fin,'%d',N);
        disp('assignments:');
        disp(assignments);
    end
    if status == 2
        N=fscanf(fin,'%d',1);
        D=fscanf(fin,'%d',1);
        fprintf(1,'params: %d %d\n',N,D);
        data=fscanf(fin,'%f',[D N])';
        disp('data:');
        disp(data);
        clf; hold on;
        plot(data(:,1),data(:,2),'.');
    end
    pause;
end
