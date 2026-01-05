def getdata(args,interp=False):
    #year 2000 => 288.46516241141825 K global mean temperature
    #year 1850 => 287.6979846532781    K global mean temperature
    #
    #a difference of 0.7671777581401216 K global mean temperature
    #
    #The enroads value is 0.79 which is roughly a 3% difference 
    #estimate 2000 from climate.gov and gisstemp 

    def monthav(x,month,ind):
        if month != 0:
            if ind==2:
                return x[:,month-1]
            else:
                return x[month-1,:,:]
        else:
            fac=np.array([31,28.25,31,30,31,30,31,31,30,31,30,31])/365.25
            if ind==2:
                y=x[:,0]*fac[0]
            else:
                y=x[0,:,:]*fac[0]
            for j in range(1,12):
                if ind==2:
                    y += x[:,j]*fac[j]
                else:
                    y += x[j,:,:]*fac[j]
            return y
        
    import h5py
    import numpy as np
    T0=287.6979846532781
    field=args["field"]
    typ=args["type"]
    if field=="precip":
        hfile=h5py.File("precip_reduced_emulator.hdf5", "r")
    else:
        hfile=h5py.File("ts_reduced_emulator.hdf5", "r")
#    lat = hfile["lat (degrees)"][:]
#    lon = hfile["lon (degrees)"][:]
    lat = hfile["lat (radians)"][:]*180.0/np.pi
    lon = hfile["lon (radians)"][:]*180.0/np.pi
    Tbar=float(args['delTbar'])
    month=int(args["month"])

    rU = hfile["modes"][:, :, :]
    mu = monthav(hfile["linear fit"][1,:,:],month,2)
#    lat = hfile["lat (degrees)"][:]
#    lon = hfile["lon (degrees)"][:]
    if "rand" in args.keys():
        sigma = monthav(hfile["monthly covariance"],month,1)
        from scipy.stats import multivariate_normal
        rand_realization=True
        seed=int(args['rand'])
        rv = multivariate_normal(mu*Tbar, sigma, seed=seed)
        realization = rv.rvs()
        v= np.tensordot(realization, rU, axes=([0], [0]))/Tbar
    else:
        v = np.tensordot(mu, rU, axes=([0], [0]))
    nr=int(len(lon)/2)
    lon=np.hstack((lon[nr:]-360,lon[0:nr+1]))
    v=np.hstack((v[:,nr:],v[:,0:nr+1]))
    if interp:
        from interpdat import interpdat
        lat_req=float(args['lat'])
        lon_req=float(args['lon'])
        vi=interpdat(lat_req,lon_req,lat,lon,v)
    else:
        vi=-999
    if typ=="del":
        return {"lat":lat,"lon":lon,"v":v*Tbar,"vi":vi*Tbar}
    if typ=="delmTbar":
        return {"lat":lat,"lon":lon,"v":v*Tbar,"vi":vi*Tbar-Tbar}
    elif typ=="minusBase":
        fac=Tbar-float(args['delBase'])
        return {"lat":lat,"lon":lon,"v":v*fac,"vi":vi*fac}
    mu0 = monthav(hfile["linear fit"][0,:,:],month,2)+T0*mu
    Tabs=np.tensordot(mu0, rU, axes=([0], [0]))
    if field=="T":Tabs -= 273.15
    Tabs=np.hstack((Tabs[:,nr:],Tabs[:,0:nr+1]))
    if interp:
        Ti=interpdat(lat_req,lon_req,lat,lon,Tabs)
    else:
        Ti=-999
    if typ=="abs":
        return {"lat":lat,"lon":lon,"v":Tabs+v*Tbar,"vi":Ti+vi*Tbar}
    sigma = monthav(hfile["monthly covariance"],month,1)
    #Tvar = np.einsum(rU, [0, 1, 2], sigma, [0, 3], rU, [3, 1, 2], [1, 2])
    vt=np.tensordot(sigma,rU,[0,0])
    Tvar=0
    nm=rU.shape[0]
    for k in range(0,nm): Tvar+=  rU[k,:,:]*vt[k,:,:]
    Tstd = np.sqrt(Tvar)
    Tstd=np.roll(Tstd,96,1)
    Tstd=np.hstack((Tstd,Tstd[:,0:1]))
    stdTi=-999.0
    if interp:
        stdTi=interpdat(lat_req,lon_req,lat,lon,Tstd)
    if typ=="std_dev":
        return {"lat":lat,"lon":lon,"v":Tstd,"vi":stdTi}
    if typ=="T90":
        return {"lat":lat,"lon":lon,"v":Tabs+v*Tbar+1.81*Tstd,"vi":Ti+vi*Tbar+1.81*stdTi}
    if typ=="Tm90":
        return {"lat":lat,"lon":lon,"v":Tabs+v*Tbar-1.81*Tstd,"vi":Ti+vi*Tbar-1.81*stdTi}
    #data
    fac=Tbar-float(args['delBase'])
    return {"lat":lat,"lon":lon,"vi":vi*Tbar,"absi":Ti+vi*Tbar,"imp":fac*vi,"std_dev":stdTi}
