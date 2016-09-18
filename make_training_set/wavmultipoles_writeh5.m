function wavmultipoles_writeh5( fname, eobdat, lidx, midx )

% Write HDF5 files from EOB output structure
%
%

LM2K = eobdat.EOBopt.LM2K;


loc = '/time/'; 
nam = 'time';

dset_details.Location = loc;
dset_details.Name = nam;

dset = eobdat.wav.t;

hdf5write(fname, dset_details, dset); 



loc = '/rwz/phase/'; 
dset_details.Location = loc;

dsetP = eobdat.wav.philm;
for l=lidx
    for m=midx
        if m>l
            continue
        end
        k = LM2K(l,m);
        nam = sprintf('phil%gm%g',l,m);
        dset_details.Name = nam;
        hdf5write(fname, dset_details, dsetP(k,:), 'WriteMode','append');
    end
end




loc = '/rwz/amplitude/'; 
dset_details.Location = loc;

dsetA = eobdat.wav.Alm;
for l=lidx
    for m=midx
        if m>l
            continue
        end
        k = LM2K(l,m);
        nam = sprintf('Al%gm%g',l,m);
        dset_details.Name = nam;
        hdf5write(fname, dset_details, dsetA(k,:), 'WriteMode','append');
    end
end



