#%%
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.ndimage.measurements import label
from skimage.measure import regionprops
import tensorflow as tf
"""
try:
    import mkl_fft as fft
except ImportError:
    """
try:
	import pyfftw.interfaces.numpy_fft  as fft
except ImportError:
	import numpy.fft as fft

#import matplotlib.patches as ptc

#%% FUNCTIONS NEEDED 
# sample covariance function (isotrope)

def im_cov(im): # image isotrope covariance function based on fft
    cov=np.real(fft.fftshift(fft.ifft2(np.absolute(fft.fft2(im))**2)))/im.size
    cov1=cov[int(np.shape(cov)[0]/2+0.5),int(np.shape(cov)[1]/2):]
    cov2=np.flip(cov[int(np.shape(cov)[0]/2+0.5),:int(np.shape(cov)[1]/2+0.5)])
    cov3=cov[int(np.shape(cov)[1]/2):,int(np.shape(cov)[0]/2+0.5)]
    cov4=np.flip(cov[:int(np.shape(cov)[1]/2+0.5),int(np.shape(cov)[0]/2+0.5)])
    cov=(cov1+cov2+cov3+cov4)/4
    lags=np.arange(np.shape(cov)[0])+0.5
    return lags, cov # x and y of the cov function   

# kmeans classification for singl/multiband images

def imkm(im,ncl,rseed=None,njobs=1): # k-means classification for multiband images
    imsum=np.sum(im,axis=np.ndim(im)-1)
    X=im[~np.isnan(imsum),:]
    # Number of clusters
    kmeans = KMeans(n_clusters=ncl,n_jobs=njobs,random_state=rseed)
    # Fitting the input data
    kmeans = kmeans.fit(X)
    # order the labels from the center closest to zero to the furtest
    c=kmeans.cluster_centers_
    csum=np.sqrt(np.sum(np.power(c,2),axis=1))
    label_ind=np.argsort(csum)
    kmeans.cluster_centers_=c[label_ind,:]
    # Getting the cluster labels
    labels = kmeans.predict(X)
    # reshape
    imth=np.zeros_like(im[:,:,0])*np.nan #%% float to support nans
    imth[~np.isnan(imsum)]=labels
    return imth

# connectivity for gategorical images
def conn(F0): # connectivity measure of the categorical image
    nan_mask=np.isfinite(F0)
    c=np.unique(F0[nan_mask]) # numerical classes
    conn0=np.empty_like(c)*np.nan
    for i in range(len(c)):
        # connectivity for F0
        l = label(F0==c[i])[0] 
        rps = regionprops(l, cache=False)
        A=[r.area for r in rps]
        conn0[i]=np.sum(np.power(A,2))/np.power(np.sum(A),2)
    return conn0

#%% GENERATION OF A MULTIGAUSSIAN FIELD
############  parameters
imsize=64 # image side length
# kernel type:
# gauss = gaussian field with gaussian covariance
# exp = gaussian field with exponential covariance (sharper variability)
ctype="gauss" #"exp" 
lc=5 # correlation length, set it between 1 and 1/10 of imsize
sigma2=2 # variance, arbitrary
###########

# field generation
x=lc*5 # kernel size, leave it to 5 times lc
h=scipy.ndimage.distance_transform_edt(np.pad(np.zeros((1,1)),(x),'constant',constant_values=1))
if ctype=="gauss":
    a=2
else:
    a=1
kernel =sigma2*np.exp(-(h/lc)**a); # kernel

# generation
simulationSize=(imsize,imsize);
sim=np.real(fft.ifft2(fft.fft2(np.random.randn(simulationSize[0],simulationSize[1]))
	*((fft.fft2(np.pad(kernel,((simulationSize[0]-kernel.shape[0],0),(simulationSize[1]-kernel.shape[1],0)))))**.5)));
plt.figure()
plt.imshow(sim)
plt.title("Gaussian Generated Image")
plt.show()

# NOTE: keep the kernel for the validation phase, since it contains the theoretical covariance


#%% IMPORT GENERATED IMAGES 


def process_image(file_path):
    img = tf.io.read_file(file_path)
    img = tf.io.decode_png(img, channels = 1, dtype = tf.dtypes.uint8)
    #img = rgb2gray(img)
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    return tf.image.resize(img, [imsize, imsize])

def create_dataset(file_path):
    list_ds = tf.data.Dataset.list_files(file_path)
    img_ds = list_ds.map(process_image)
    for image in img_ds.take(1):
        print("Image shape:", image.numpy().shape)
    img_ds = img_ds.as_numpy_iterator()
    img_ds = np.array(list(img_ds))
    return img_ds

generated_images = create_dataset('../GeneratedImages/GAN/*.png')

#%% QUALITY INDICATORS:
#% 1) HISTOGRAM

# fake series of 10 simulations, replace with your images generated with MPS/GAN
print("Simulation shape : ", sim[:, :, None].shape)
#rsim=np.repeat(sim[:, :, None],10,axis=2)
#print(rsim.shape)

rsim = generated_images[0:10].reshape(64,64,10)
plt.figure()
plt.axis('off')
plt.imshow(rsim[1].squeeze())
#rsim = X_train
rsim=rsim+np.random.rand(np.shape(rsim)[0],np.shape(rsim)[1],np.shape(rsim)[2])
#%%
#histogram comparison
yref,x=np.histogram(sim.ravel())
x=(x[:-1]+x[1:])/2
y=np.zeros((len(x),np.shape(rsim)[2]))
for i in range(np.shape(rsim)[2]):
    y[:,i],xtmp=np.histogram(rsim[:,:,i])#.ravel())
plt.figure()
plt.boxplot(np.rot90(y),positions=x,manage_ticks=False)
plt.plot(x,yref,"-o",label="reference")
plt.legend()
plt.show()

#%% 2) SAMPLE COVARIANCE
# retrive central line from kernel, used to generate th ref fields, as theoretical covariance function
cov_ref=np.copy(kernel[int(np.shape(kernel)[0]/2+0.5),int(np.shape(kernel)[1]/2+1):])
# and its lags
lags_ref=np.arange(np.shape(cov_ref)[0])+0.5

# sample covariance, to compute on fields simulated using MPS/GAN, 
# here we compute it on the ref field, 
# you can show both theoretical and sample covariance for reference fields

lags,cov=im_cov(sim)

# plot
plt.figure()
plt.plot(lags,cov,label="sample covariance") # sample covariance
plt.plot(lags_ref,cov_ref,label="reference covariance") # kernel
plt.legend()
plt.show()

#%% 3) Kmeans classification conncetivity (5 classes)
ncl=5 #number of classes
sim_km=imkm(sim[:,:,None],ncl) # kmeans classification of the reference image
plt.figure() # show classification
plt.imshow(sim_km)
plt.show()

# connectivity measure
sim_kmcc=conn(sim_km) # connectivity measure for each class (probability of pixels to be connceted)
x=np.arange(ncl) # number of classes on the x axis
# km connectivity for all simulations (rsim matrix)
y=np.zeros((len(x),np.shape(rsim)[2]))
for i in range(np.shape(rsim)[2]):
    imtmp=np.squeeze(rsim[:,:,i])
    rsim_km=imkm(imtmp[:,:,None],ncl)
    y[:,i]=conn(rsim_km)
    print(i,"/",np.shape(rsim)[2]-1)
    
#% show conncetivity for all classes, reference image, and one simulation 
plt.figure(figsize=(8,3))
plt.subplot(1,3,1)
plt.boxplot(np.rot90(y),positions=x,manage_ticks=False)
plt.plot(x,sim_kmcc,"-o",label="reference")
plt.xlabel("classes")
plt.ylabel("conncetivity [0-1]")
plt.legend()
plt.subplot(1,3,2)
plt.imshow(sim_km)
plt.colorbar()
plt.title("reference")
plt.subplot(1,3,3)
plt.imshow(rsim_km)
plt.colorbar()
plt.title("one simulation")
plt.tight_layout()
plt.show()

