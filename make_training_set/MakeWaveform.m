function MakeWaveform(fname, nu, ka2, ka3, ka4, kb2, kb3, kb4)
%MakeWaveform Generate the tidal EOB waveform.

nu2 = nu^2;

% set up the Tidal structure

% tidal pars
Topt.kappaA = [0 ka2 ka3 ka4];
Topt.kappaB = [0 kb2 kb3 kb4];

Topt.TidalUse  = 'yes';
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
%r0 = ( 0.01/2 )^(-2/3); 

% Value for random comparison waveforms:
%r0=30.0

% My original ~10Hz value (which is wrong):
%r0 = 218.0

% Correct 10Hz value:
r0 = 230.0

% final time
%tmax = 1e8;
tmax = 1e10;
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

wavmultipoles_writeh5( [fname,'.h5'], out, 2,2 );
%wavmultipoles_writeh5( [fname,'.h5'], out, [2:6], [1:6] );
system(['gzip ',fname,'.h5']);


end

