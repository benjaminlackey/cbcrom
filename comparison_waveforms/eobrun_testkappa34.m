% Script for EOB runs
% Test systematic uncertainty on the kappas due to 
% - the \Lambda_3,4(\Lambda_2) fits
% - the l=3,4 terms

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EOB code (path to your EOB dir)
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%addpath ../../eobcode/MatlabCode_20160408
addpath /home/bdlackey/teob_surrogate/eobcode/MatlabCode_20160408/
addpath /home/bdlackey/cbcrom/make_training_set
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Configurations to test
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

i=0;

% -------------------------------------------------------------
% BGN1H1 target
i=i+1; 
parspace(i).('name') = 'BGN1H1_target';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 1.3134375000000003 1.1705859375000003 1.3803808593750002 ];
parspace(i).('kapb') = [0 1.3134375000000003 1.1705859375000003 1.3803808593750002 ];

% BGN1H1 fit
i=i+1; 
parspace(i).('name') = 'BGN1H1_fit';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 1.3134375000000003 0.97334476664527425 0.98395515300952419 ];
parspace(i).('kapb') = [0 1.3134375000000003 0.97334476664527425 0.98395515300952419 ];

% -------------------------------------------------------------
% MPA1 target
i=i+1; 
parspace(i).('name') = 'MPA1_target';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 0.53475000000000006 0.28125000000000006 0.19394238281250004 ];
parspace(i).('kapb') = [0 0.53475000000000006 0.28125000000000006 0.19394238281250004 ];

% MPA1 fit
i=i+1; 
parspace(i).('name') = 'MPA1_fit';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 0.53475000000000006 0.31036979381671392 0.23834389540412038 ];
parspace(i).('kapb') = [0 0.53475000000000006 0.31036979381671392 0.23834389540412038 ];

% -------------------------------------------------------------
% sly 1.35+1.35 target
i=i+1; 
parspace(i).('name') = 'sly135135_target';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 36.522889624695843 82.609133619691761 253.81639025942832];
parspace(i).('kapb') = [0 36.522889624695843 82.609133619691761 253.81639025942832];

% sly 1.35+1.35 fit
i=i+1; 
parspace(i).('name') = 'sly135135_fit';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 36.522889624695843 80.782544466956168 257.86687968726704];
parspace(i).('kapb') = [0 36.522889624695843 80.782544466956168 257.86687968726704];

% sly 1.35+1.35 k4=0
i=i+1; 
parspace(i).('name') = 'sly135135_k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 36.522889624695843 82.609133619691761 0];
parspace(i).('kapb') = [0 36.522889624695843 82.609133619691761 0];

% sly 1.35+1.35 k3=k4=0
i=i+1; 
parspace(i).('name') = 'sly135135_k3k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 36.522889624695843 0 0];
parspace(i).('kapb') = [0 36.522889624695843 0 0];

% -------------------------------------------------------------
% MS1b 1.35+1.35 target
i=i+1; 
parspace(i).('name') = 'ms1b135135_target';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 143.64460264508691 527.40645651795592 2627.4588087291813];
parspace(i).('kapb') = [0 143.64460264508691 527.40645651795592 2627.4588087291813];

% MS1b 1.35+1.35 fit
i=i+1; 
parspace(i).('name') = 'ms1b135135_fit';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 143.64460264508691 526.88846146201547 2823.3700715988198];
parspace(i).('kapb') = [0 143.64460264508691 526.88846146201547 2823.3700715988198];

% MS1b 1.35+1.35 k4=0
i=i+1; 
parspace(i).('name') = 'ms1b135135_k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 143.64460264508691 527.40645651795592 0];
parspace(i).('kapb') = [0 143.64460264508691 527.40645651795592 0];

% MS1b 1.35+1.35 k3=k4=0
i=i+1; 
parspace(i).('name') = 'ms1b135135_k3k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 143.64460264508691 0 0];
parspace(i).('kapb') = [0 143.64460264508691 0 0];

% -------------------------------------------------------------
% MS1b 1.4+1.4 
%TOVL(7.16274e-04,eos(1),[2 3 4],[1e-9:(10-1e-9)/1000:10],0); % new TOVL.m
% c =1.470206825706509e-01
% k2 = 1.298764509934916e-01
% k3 = 3.830027378466424e-02
% k4 = 1.531642180986093e-02
% blam2 = 1260.5119868750148  
% blam3 = 3439.4703670512176 blam3_yagi = 3437.3257215172985    
% blam4 = 9090.5906603366293 blam4_yagi = 9764.9680362720555
% kappa3_yagi = 402.81160799030852
% kappa4_yagi = 2002.5813355636053
i=i+1; 
parspace(i).('name') = 'ms1b140140_target';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 1.181729987695327e+02 4.030629336388147e+02 1.864281287764348e+03];
parspace(i).('kapb') = [0 1.181729987695327e+02 4.030629336388147e+02 1.864281287764348e+03];

i=i+1; 
parspace(i).('name') = 'ms1b140140_fit';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 1.181729987695327e+02 402.81160799030852 2002.5813355636053];
parspace(i).('kapb') = [0 1.181729987695327e+02 402.81160799030852 2002.5813355636053];

i=i+1; 
parspace(i).('name') = 'ms1b140140_k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 1.181729987695327e+02 4.030629336388147e+02 0];
parspace(i).('kapb') = [0 1.181729987695327e+02 4.030629336388147e+02 0];

i=i+1; 
parspace(i).('name') = 'ms1b140140_k3k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 1.181729987695327e+02 0 0];
parspace(i).('kapb') = [0 1.181729987695327e+02 0 0];

% -------------------------------------------------------------
% Sly 1.4+1.4 
%TOVL(1.43504e-03,eos(2),[2 3 4],[1e-9:(8-1e-9)/2000:8],0); % new TOVL.m
% c = 1.804707281418182e-01
% k2 = 8.803710210210396e-02
% k3 = 2.376308553421779e-02
% k4 = 8.738232353510780e-03
% blam2 = 306.57774578409862 
% blam3 = 508.15262548806874 blam3_yagi = 497.73608286328886
% blam4 = 819.60264261117015 blam4_yagi = 831.01680382559994
% kappa3_yagi = 58.328447210541675
% kappa4_yagi = 170.42336797204689
i=i+1; 
parspace(i).('name') = 'sly140140_target';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 2.874166366725927e+01 5.954913579938313e+01 1.680825731917441e+02];
parspace(i).('kapb') = [0 2.874166366725927e+01 5.954913579938313e+01 1.680825731917441e+02];

i=i+1; 
parspace(i).('name') = 'sly140140_k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 2.874166366725927e+01 5.954913579938313e+01 0];
parspace(i).('kapb') = [0 2.874166366725927e+01 5.954913579938313e+01 0];

i=i+1; 
parspace(i).('name') = 'sly140140_k3k40';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 2.874166366725927e+01 0 0];
parspace(i).('kapb') = [0 2.874166366725927e+01 0 0];

i=i+1; 
parspace(i).('name') = 'sly140140_fit';
parspace(i).('q') = 1.;
parspace(i).('nu') = 1/4;
parspace(i).('kapa') = [0 2.874166366725927e+01 58.328447210541675 170.42336797204689];
parspace(i).('kapb') = [0 2.874166366725927e+01 58.328447210541675 170.42336797204689];


% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Get down to work
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n = length(parspace);
fprintf('*** %d configurations \n',n);

for i=1:n
    
tic; 
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % EOB setting
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % mass ratio/sym mass ratio
    q   = parspace(i).('q');
    nu  = parspace(i).('nu');
    nu2 = nu^2;
    
    % set up the Tidal structure
    
    % tidal pars
    Topt.kappaA = parspace(i).('kapa');
    Topt.kappaB = parspace(i).('kapb');
    
    Topt.TidalUse  = 'yes';
    %Topt.TidalUse  = 'no';
    Topt.PNTidal   = 'nnlo_gsfLR';
    
    if nnz(Topt.kappaA)==0 & nnz(Topt.kappaB)==0
        Topt.TidalUse  = 'no';
    end
   
    Topt.ppar = 4;      % Bini & Damour, [1409.6933] p parameter
    Topt.rLRpar = [];
    Topt.b3  = [0 0 0]; % 3PN formal correction (unused)

    % 4PN and 5PN effective corrections
    ainput{5} = 0; % use 4PN analytical
    ainput{6} = 3097.3*nu2 - 1330.6*nu + 81.38; % 5PN
        
    % initial radius
    r0 = ( 0.001/2 )^(-2/3); 
    
    % final time
    tmax = 1e8;
    %tmax = 1e10;
    dt   = 1;
    
    % output mode
    modus   = 1; % (1=> inspl) (2=> inspl+mrg) (3=> inspl+mrg+rng)
    verbose = 0; % 0 = minimum, 1 = text ,2 = text+figs
    saveme  = 0; % save data ?
    
    % options structure
    options = EOBSet('Dynamics','eob',...
    'RadReac','din',...
    'ddotrMethodDyn','noflux',...
    'ddotrMethodWav','noflux',...
    'HorizonFlux','no',...
    'RadialFlux','no',...
    'FIterms','yes',...
    'rholmarg','v_phi',...
    'PNorder','5pnP15',...
    'resumD','pade03',...
    'resumf22','no',...
    'Tidal',Topt.TidalUse,...
    'TidalStruct',Topt,...
    'NewtonWaves','no',...
    'UseNQC','no',...
    'DetermineNQCab','no',...
    'NQCFileNR','./DataNR/NRpts4NQC.dat',...
    'QNMDataDir','./dataQNMs/',....
    'textend',0,...
    'ODETime','nonuniform',...
    'ODEEventFunRmin',1,...
    'ODEEventFunAeps',1e-9,...
    'ODESolverRelTol',1e-11,...
    'ODESolverAbsTol',1e-14,...
    'ODESolverRefine',4); 
    
    % following pars are required but not used    
    % NQC a_i's for wave/radiation reaction
    LM2K = Multipidx(2);
    ainput{1} = zeros(1,length(LM2K));
    ainput{2} = ainput{1};
    ainput{3} = ainput{1};
    ainput{4} = ainput{1};

    % QNM
    nQNM   = [2 0];
    lmaxQNM = 2;

    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % EOB run
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    out = EOBRun(nu, ainput, r0,tmax,dt, nQNM,lmaxQNM, ...
        options,...
        modus, verbose, saveme);
    
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%    
    % write hdf5 data
    % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    wavmultipoles_writeh5( [parspace(i).('name'),'.h5'], out, 2,2 );
    %wavmultipoles_writeh5( [fname,'.h5'], out, [2:6], [1:6] );
    system(['gzip ',parspace(i).('name'),'.h5']);
    
    fprintf('done %s s\n',toc)
          
end

