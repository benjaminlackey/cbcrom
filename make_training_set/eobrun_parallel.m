% Parameters uniform in Lambda_a, Lambda_b

% Script for EOB runs

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EOB code
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath /Users/lackey/Research/TidalEOBMatlab/MatlabCodePolished/
%addpath /home/bdlackey/teob/MatlabCodePolished/
addpath /home/bdlackey/teob_surrogate/eobcode/MatlabCode_20160408/

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load parspace file
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parspace = load('parspace_smallq_cheb_3.out');
%parspace = load('parspace_smallq_cheb_9.out');
parspace = load('parspace_smallq_cheb_16.out');
%parspace = load('parspace_rand.out');


n = size(parspace,1);
fprintf('*** %d configurations \n',n);

% LambdaA,LambdaB,SymMassRatio,MassRatio,kappaA2,kappaB2,kappaA3,kappaB3,kappaA4,kappaB4
col_La  = 1;
col_Lb  = 2;
col_nu  = 3;
col_q   = 4;
col_ka2 = 5;
col_kb2 = 6;
col_ka3 = 7;
col_kb3 = 8;
col_ka4 = 9;
col_kb4 = 10;

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get down to work
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outdir = '../training_set_cheb_3/'
%outdir = '../training_set_cheb_9/'
outdir = '../training_set_cheb_16/'
%outdir = '../training_set_rand/'

nnodes = 6;
%matlabpool('open', 'cluster', nnodes);
matlabpool('open', nnodes);
%parpool(4);
%for i=1:n
%parfor i=1:n
%parfor i=1:1000
parfor i=1:2000
    tic; 
    sss = sprintf('%.6f_%.6f_%.6f',...
        [parspace(i,col_q) parspace(i,col_La) parspace(i,col_Lb)]);
    sss = strrep(sss,'.','p'); 
    fname = [outdir,'teob_',sss];
    fprintf('===> %d : %s\n',i,sss);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EOB setting
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % mass ratio/sym mass ratio
    nu  = parspace(i,col_nu);
    ka2 = parspace(i,col_ka2);
    ka3 = parspace(i,col_ka3);
    ka4 = parspace(i,col_ka4);
    kb2 = parspace(i,col_kb2);
    kb3 = parspace(i,col_kb3);
    kb4 = parspace(i,col_kb4);
    MakeWaveform(fname, nu, ka2, ka3, ka4, kb2, kb3, kb4)
    
    fprintf('done %s s\n',toc)
end
matlabpool close;
% Delete the current pool that you get with 'get current parallel pool'
%delete(gcp);
