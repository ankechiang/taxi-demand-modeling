# taxi-demand-modeling


## 1. Reference
+ Meng-Fen Chiang, Tuan-Anh Hoang, Ee-Peng Lim:, ["Where are the passengers?: a grid-based gaussian mixture model for taxi bookings."](http://ink.library.smu.edu.sg/cgi/viewcontent.cgi?article=4171&context=sis_research) SIGSPATIAL/GIS 2015

## 2. Input Format
+ Each booking is put in a file. Each line in the file is of the format:  
  ...  
  <date_i>,<time-of-a-day_i>,<latitude_i>,<longitude_i>,<lat-grid-id_i>,<lon-grid-id_i>  
  ...  
  e.g., 2014-05-06,503.0,1.321490,103.908600,7,34  
+ File path: rootdir/grabTaxi/input/sg_bookings_jul_sep.txt_wday_b100.csv

## 3. Output Format
+ Loglikelihood: rootdir/grabTaxi/output/logLik(k1)_(k2)_wday_jul_sep_d(time-slot).csv  
where (k1), (k2) and (time-slot) are smallest num K, largest num K and time window of interest, which are specified during running the model  
+ Parameters: rootdir/grabTaxi/output/GGMM/ggmm_(day)_jul_sep_k_w(t1)_w(t2)
where (t1)/(t2) is the starting/ending time windows, which is specified during running the model. 
  * mixture weights of K Gaussians of each cell  
  * mu of K Gaussians of each cell  
  * covariance matrix of K Gaussians of each cell  
    e.g.,  
    ...  
    pi_0=[ 0.294605809129 0.634854771784 0.0705394190871 ]  
    clusters = 3  
    mu_0=[ 1.30096634019 103.828820661 360.28973179 ]  
    sigma_0=[ [4.060461561835867e-05, 0.0, 0.0] [0.0, 0.0046409619060315144, 0.0] [0.0, 0.0, 438.4028687300076] ]  
    ...  
