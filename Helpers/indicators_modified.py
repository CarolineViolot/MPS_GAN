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
print(sim.shape)
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



#%% QUALITY INDICATORS:
#% 1) HISTOGRAM

# fake series of 10 simulations, replace with your images generated with MPS/GAN
#print("Simulation shape : ", sim[:, :, None].shape)
rsim=np.repeat(sim[:, :, None],10,axis=2)
#print(rsim.shape)

#rsim = generated_images[0:400].reshape(64,64,400)

rsim=rsim+np.random.rand(np.shape(rsim)[0],np.shape(rsim)[1],np.shape(rsim)[2])

#%%
#histogram comparison

#selection of generated images at the end of the training ('good images')
generated_images = create_dataset('../GeneratedImages/GAN/Gaussian64x64/NICE/*.png')[:10]#.reshape(64,64,10)
print(generated_images.shape)
real_images = create_dataset('../Datasets/Gaussian64x64/Images/image10*.png')[:10]

#%%

yref,x=np.histogram(real_images[0].ravel())
x=(x[:-1]+x[1:])/2
#plt.plot(x, yref)

#plt.hist(real_images[0].ravel())
y, x = np.histogram(generated_images[0].ravel())
x=(x[:-1]+x[1:])/2
plt.plot(x, y)
print(np.shape(real_images[0].ravel()))
print(np.shape(generated_images[0].ravel()))
print(np.shape(generated_images[0]))
print(np.shape(real_images[0]))
#%%
#real_image_mean = real_images[0]/10
#for i in range (1, 10): 
#real_image_mean = real_image_mean+real_images[i]/10

plt.figure(1)
plt.imshow(generated_images[0].squeeze())
plt.figure(2)
plt.imshow(real_images[0].squeeze())

yref,x=np.histogram(real_images[1].ravel())
x=(x[:-1]+x[1:])/2


y=np.zeros((len(x),np.shape(generated_images)[2]))

for i in range(0, 10):
    #print(i)
    y[:,i],xtmp=np.histogram(generated_images[i])#.ravel())
    print(y[:,i])

plt.figure()
plt.boxplot(np.rot90(y),positions=x,manage_ticks=False, widths=0.05)

plt.plot(x, np.mean(y, 1))
plt.plot(x,yref,"-o",label="reference")
plt.xlim((0, 1))
plt.legend()
plt.show()

#%% 3) Kmeans classification conncetivity (5 classes)
ncl=10 #number of classes
sim_km=imkm(real_images[0].squeeze()[:,:,None],ncl) # kmeans classification of the reference image
plt.figure() # show classification
plt.imshow(sim_km)
plt.show()

rsim = generated_images
plt.figure()
plt.imshow(generated_images[9].squeeze())
# connectivity measure
sim_kmcc=conn(sim_km) # connectivity measure for each class (probability of pixels to be connceted)
x=np.arange(ncl) # number of classes on the x axis
# km connectivity for all simulations (rsim matrix)
y=np.zeros((len(x),np.shape(rsim)[2]))
for i in range(np.shape(rsim)[0]):
    imtmp=np.squeeze(rsim[i].squeeze())
    print(imtmp.shape)
    rsim_km=imkm(imtmp[:,:,None],ncl)
    y[:,i]=conn(rsim_km)
    print(i,"/",np.shape(rsim)[2]-1)
 
#rsim_km = imkm(generated_images[0].squeeze()[:,:,None],ncl)
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

