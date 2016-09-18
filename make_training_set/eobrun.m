function eobrun(La, Lb, nu, q, ka2, kb2, ka3, kb3, ka4, kb4)

La = str2num(La)
Lb = str2num(Lb)
nu = str2num(nu)
q = str2num(q)
ka2 = str2num(ka2)
kb2 = str2num(kb2)
ka3 = str2num(ka3)
kb3 = str2num(kb3)
ka4 = str2num(ka4)
kb4 = str2num(kb4)

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EOB code
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath /Users/lackey/Research/TidalEOBMatlab/MatlabCodePolished/
%addpath /home/bdlackey/teob/MatlabCodePolished/
%addpath /home/bdlackey/teob_surrogate/eobcode/MatlabCode_20160408/

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load parspace file
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%parspace = load('parspace_smallq_cheb_3.out');
%parspace = load('parspace_smallq_cheb_9.out');
%parspace = load('parspace_smallq_cheb_16.out');
%parspace = load('parspace_rand.out');


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get down to work
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%outdir = '../training_set_cheb_3/'
%outdir = '../training_set_cheb_9/'
%outdir = '../training_set_cheb_16/'
%outdir = '../training_set_rand/'
outdir = '../training_set_test/';

tic; 
sss = sprintf('%.6f_%.6f_%.6f',[q La Lb]);
sss = strrep(sss,'.','p'); 
fname = [outdir,'teob_',sss];
fprintf('fname: %s\n', fname);

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EOB setting
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

MakeWaveform(fname, nu, ka2, ka3, ka4, kb2, kb3, kb4)

fprintf('done %s s\n',toc)

end
