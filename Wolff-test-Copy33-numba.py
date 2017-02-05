#from gevent import monkey
#monkey.patch_all()
#import gevent
#from gevent import pool

import base64
try:
    base64.encodebytes
except AttributeError:
    base64.encodebytes = base64.encodestring

# In[2]:

from pylab import *
import pandas as pd
import itertools as it
from joblib import Parallel, delayed
from matplotlib import animation as an
#from IPython import display
from tempfile import NamedTemporaryFile


# ## Video emedding

# In[3]:

from numba import jit,b1, u2, i8, void
#import numba as nb

#@jit #(b1(b1[:,:], i8[:])) #, nopython=True)
def aget(lattice, coords):
    return lattice.item(*coords.tolist())

#@jit #(void(b1[:,:], i8[:], b1)) #, nopython=True)
def aset(lattice, coords, val):
    l = coords.tolist()
    l.append(val)
    lattice.itemset(*l)

@jit(nopython=True)
def nextpos(current, dims, dim, delta):
    nxt = current.copy()
    nxt[dim] = (nxt[dim]+delta)%dims[dim]
    return nxt

#@jit#(nopython=True)
def cluster(lattice, prob):
    dims = np.array(lattice.shape)
    start = mlab.amap(randint,dims)
    state = not aget(lattice, start)
    aset(lattice,start,state)
    #print >>sys.stderr, "$$$",
    ndim = lattice.ndim
    n = 0
    S = [start]
    while S:
        n+=1
        current = S.pop()
        for dim in range(ndim):
            for delta in [-1,1]:
                nxt = nextpos(current, dims, dim, delta)
                if  (np.random.random() < prob) and (aget(lattice,nxt) ^ state):
                    aset(lattice,nxt,state)
                    #print >>sys.stderr, nxt,
                    S.append(nxt)
    #print >>sys.stderr
    return n*(2*state-1)

#@jit
def run(lattice, prob):
    yield lattice.sum() - lattice.size/2
    while True:
        yield cluster(lattice, prob)


# ## Finite lattice corrrections

# In[6]:

Tc = 2/log(1 + sqrt(2))
Pc = 1 - 1/(1+sqrt(2))
#xi = .3603 ## value given by ...
Xi = 2 # value that actually seems optimal here

def Tfin(dim, xi=Xi):
    return Tc*(1 + float(xi)/dim)

def Prob(T):
    return 1-exp(-2/T)


# In[7]:

deltas0=[]
d = 2**8
A = zeros((d,d), dtype=bool)
#A = np.random.random((d,d)) < 0.5
Pfin = Prob(Tfin(d))
print Pfin, A.size
print Tc, Pc, Prob(Tc), Tfin(d), Prob(Tfin(d))


# In[ ]:


############################

# deltas0=fromiter(run(A, Pfin), int, A.size)
# exit(0)

###########################################################

# def nrun(xi):
#     AA = zeros((d,d), dtype=bool)
#     return fromiter(run(AA, Prob(Tfin(d, xi))), int, 2*AA.size)

# deltas = pd.DataFrame(Parallel(n_jobs=-1, verbose=3)(amap(delayed(nrun), linspace(0, 2.0, 201)))).T

# pd.set_option("display.width", 256)

# print abs(deltas).describe()

# print abs(deltas.cumsum()).describe()

# print (deltas.cumsum()**2).describe()

# deltas.to_csv("zzz2.csv")
# exit(0)

###########################################################


from sklearn import decomposition as dec
def cmpr(a, k):
    model = dec.NMF(k)
    print model
    print a
    print ma.is_masked(a)
    a = a.astype(float)
    print a
    print ma.is_masked(a)
    a=model.fit_transform(a)
    print a
    print ma.is_masked(a)
    return model.inverse_transform(a)
    #return a.dot(model.components_)

from scipy import ndimage
from scipy import signal as sig
from skimage import restoration as rst

#@jit(nopython=True)
def smooth(a):
    a = a.astype(float)
    #orig = a
    a = (a + rand(*a.shape))/2
    #a = a + randint(0,2,a.shape)
    #a = a + normal(0,2**-1,a.shape)
    #a = a+standard_cauchy(a.shape)
    #a = ndimage.median_filter(a, 5, mode='wrap')
    #a = ndimage.uniform_filter(a, 2**1, mode='wrap')
    #a = ndimage.fourier_uniform(a, 2**5)
    #a = 1.1**a
    #a = cmpr(a, 2**5)
    #a = fftcmp2(a, 2**12)
    #a=denoise(a, 2**-0.5, 'bior3.1')
    a=denoise(a, 2**-0.5, 'bior3.1')
    #a=denoise(a, 2**-1.5, 'sym3')
    #a = rst.denoise_bilateral(a, 2**5)
    #a = sig.wiener(a,6)
    #a = ndimage.median_filter(a, 2**1, mode='wrap')
    #a = sig.hilbert2(a)
    #err = a - orig
    #print >>sys.stderr, a.min(), a.max(), a.mean(), a.std(), orig.std(), sqrt((err**2).mean())
    #print >>sys.stderr, abs(orig - orig.mean()).mean(), abs(a - a.mean()).mean(), abs(err).mean()
    #a -= a.mean()
    #a -= 0.15
    #a /= a.std()
    return a

import pywt

from sklearn import preprocessing as pre

#@jit(nopython=True)
def denoise(data, noiseSigma, wavenm, mode='ppd'):
    wavelet = pywt.Wavelet(wavenm)
    WC = pywt.wavedec2(data,wavelet,mode=mode)
    #print wavenm
    #print data.mean(), data.std(), data.min(), data.max()
    threshold=noiseSigma*sqrt(2*log2(data.size))
    WC = [WC[0]] + map(lambda x: pywt.thresholding.soft(x,threshold), WC[1:])
    #WC = [WC[0]] + map(lambda x: pywt.thresholding.hard(x,threshold), WC[1:])
    #WC = [WC[0]] + map(lambda x: pywt.thresholding.greater(x,-threshold,-threshold), WC[1:])
    #WC = [WC[0]] + map(lambda x: pywt.thresholding.less(x,threshold,threshold), WC[1:])
    #WC = map(lambda x: pywt.thresholding.soft(x,threshold), WC)
    #WC = map(lambda x: pywt.thresholding.hard(x,threshold), WC)
    #WC = [WC[0]]+map(lambda x: (abs(asarray(x, float64))**noiseSigma) * sign(x), WC[1:])
    #WC = map(lambda x: (abs(asarray(x, float64))**noiseSigma) * sign(x), WC)
    #ZZZ = hstack(map(lambda x: ravel(asarray(x)), NWC))
    #print >>sys.stderr, ">>>", ZZZ.mean(), ZZZ.std(), ZZZ.min(), ZZZ.max()
    #print >>sys.stderr, len(NWC), map(lambda x: x.shape, NWC)
    #print WC
    #print NWC
    res = pywt.waverec2(WC, wavelet, mode=mode)
    #print res.mean(), res.std(), res.min(), res.max()
    #print
    return pre.minmax_scale(pywt.waverec2(WC, wavelet, mode=mode).ravel()).reshape(data.shape)


from scipy import fftpack
def get_2D_dct(img):
    """ Get 2D Cosine Transform of Image
    """
    return fftpack.dct(fftpack.dct(img.T, norm='ortho').T, norm='ortho')

def get_2d_idct(coefficients):
    """ Get 2D Inverse Cosine Transform of Image
    """
    return fftpack.idct(fftpack.idct(coefficients.T, norm='ortho').T, norm='ortho')

def fftcmp(pixels, ii):
    dct_size = pixels.shape[0]
    dct = get_2D_dct(pixels)
    reconstructed_images = []
    dct_copy = dct.copy()
    dct_copy[ii:,:] = 0
    dct_copy[:,ii:] = 0
    return get_2d_idct(dct_copy);

def fftcmp2(pixels, k):
    dct = get_2D_dct(pixels)
    mask = ((-abs(dct.ravel())).argsort().argsort() < k).reshape(dct.shape)
    dct2 = dct*mask.astype(float)
    idct = get_2d_idct(dct2)
    return idct

from skimage.util.shape import view_as_blocks


def blocks(img):
    block_shape = (2, 2)
    view = view_as_blocks(img, block_shape)
    flatten_view = view.reshape(view.shape[0], view.shape[1], -1)
    sum_view = np.sum(flatten_view, axis=2)
    print >>sys.stderr, log2(bincount(sum_view.ravel())+1)
    mean_view = sum_view/float(flatten_view.shape[-1])
    return (2*mean_view - 1)# ** 3


import sys
#@jit
def animate(img, prob, frames=60):
    lattice = img.get_array()
    def init():
        img.set_array(lattice)
        return img
    def frm(i):
        cluster(lattice, prob)
        img.set_array(lattice)
        print >>sys.stderr, i,
        return img
    return an.FuncAnimation(img.get_figure(), init_func=init, func=frm, frames=frames, repeat=False, blit=False, interval=2**-20)

def animate2(lattice, prob, frames=60):
    img = imshow(blocks(lattice), cmap="gray",
             vmin=-1., vmax=1.)
    def init():
        img.set_array(blocks(lattice))
        return img
    def frm(i):
        cluster(lattice, prob)
        img.set_array(blocks(lattice))
        print >>sys.stderr, i,
        return img
    return an.FuncAnimation(img.get_figure(), init_func=init, func=frm, frames=frames, repeat=False, blit=False, interval=2**-20)



r, c = blocks(A).shape
dpi = gcf().dpi
close()
figure(figsize=(r/dpi, c/dpi))

# Img=figimage(A, cmap="gray",
#              vmin=-0., vmax=1.)

#Img=imshow(A,cmap="gray", vmin=0.0, vmax=1.0)

# print dir(Img)
crf=16
fps=18
secs = 4*60
nfrm = secs*fps
fs = 2**17
bv = "14k"
anim = animate2(A, Pfin, 2**20)
show()

# with NamedTemporaryFile(dir='.', prefix='isingwolff-d%s-xi%s-crf%s-fps%s-' % (d,Xi,crf, fps), suffix='.mp4', delete=False) as f:
#     anim.save(f.name, fps=fps, dpi=Img.get_figure().get_dpi(), extra_args=['-vcodec', 'libx264', "-crf", str(crf)], writer='ffmpeg')

# with NamedTemporaryFile(dir='.', prefix='isingwolff-d%s-xi%s-nfrm%s-fs%s-' % (d, Xi, nfrm, fs), suffix='.mp4', delete=False) as f:
#     anim.save(f.name, fps=fps, dpi=Img.get_figure().get_dpi(), extra_args=['-vcodec', 'libx264', "-fs", str(fs)], writer='ffmpeg')

# with NamedTemporaryFile(dir='.', prefix='isingwolff-d%s-xi%s-secs%s-bv%s-fps%s-' % (d, Xi, secs, bv, fps), suffix='.mp4', delete=False) as f:
#     anim.save(f.name, fps=fps, dpi=Img.get_figure().get_dpi(), extra_args=['-vcodec', 'libx264', "-b:v", bv], writer='ffmpeg')

exit(0)

###########################################################


print isinteractive()
grp = pool.Pool(100)

#imgx=figimage(A, cmap="gray")
ZZ=0
#@jit
#def vis():
    #figure()
    #fig, ax = plt.subplots()
    #figure(figsize=(18,18))
def vis(x):
        global ZZ, imgx
        if ZZ == 0:
           print ".",
           A[0,0] = 1
           # delaxes()
           #ion()
           imgx=figimage(A, cmap="gray")
           pause(2**-32)
           #fig = plt.gcf()
           #draw()
           # ax = plt.Axes(fig, [0., 0., 1., 1.])
           # ax.set_axis_off()
           # fig.add_axes(ax)
           #axis("off")
           #show()
           #clim()   # clamp the color limits
           #title("Boring slide show")
        else:
           pass
           imgx.set_array(A)
           #draw()
           #gevent.spawn(lambda : imgx.set_data(A)).start()
           #suptitle("   ".join(map(str,[ZZ, x, A.sum()])))
           if ZZ % 5 == 0: pause(10**-30)
           #if ZZ % 5 == 0: grp.apply(pause,[2**-32])
           #gevent.spawn(lambda : pause(2**-32)).start()
        ZZ+=1
        return x
#return vis2


# In[ ]:

#figure(figsize=(18,18))
#axis("off")





deltas0=fromiter(it.imap(vis, run(A, Prob(Tfin(d)))), int, A.size)

exit(0)

# In[ ]:

#%time anim=animate(A,Pfin,deltas0.append,d**2)
#%time display_animation(anim)


# In[ ]:

deltas=array(deltas0)
sig=cumsum(deltas)
fluct=abs(deltas)

pwr=abs(sig) # + A.size/2
#plot(signal.medfilt(sig/(d**2.),1))
#plot(pwr)


# In[ ]:

figure(figsize=(5, 5))

plot(sorted(log2(fluct), reverse=True))


# In[ ]:

plot(log2(arange(len(pwr))), log2(sorted((pwr), reverse=True)))


# In[ ]:

plot(log2(bincount(fluct/(d))))


# In[ ]:

flucttbl=bincount(fluct/(d))
figure(figsize=(7,7))
plot(log2(arange(len(flucttbl))+0), log2(flucttbl))


# In[ ]:

pwrtbl=bincount((pwr/d))
plot((arange(pwrtbl.size)), (pwrtbl))


# In[ ]:

figure(figsize=(15,3))
plot(deltas[-1000:])


# In[ ]:

figure(figsize=(15,3))
plot(sig[-1000:]**2)


# In[ ]:

figure(figsize=(7,7))
scatter(arcsinh((sig[:-1])/float(d**1.3)),arcsinh((sig[1:])/float(d**1.3)),alpha=2**-4, marker=".")


# In[ ]:

figure(figsize=(7,7))
scatter(arcsinh(deltas[1:]/float(d)),arcsinh(deltas[:-1]/float(d)),alpha=2**-4, marker=".")


# In[ ]:

figure(figsize=(7,7))
scatter((sig/float(d**0)),(deltas/float(d**0)),alpha=2**-3, marker=".")


# In[ ]:

from scipy import signal
#plot(log2(arange(len(deltas))+1), signal.medfilt((abs(cumsum(deltas))), 8*d-1))


# In[ ]:

#plot(signal.medfilt((fluct), 8*d-1))


# In[ ]:

#plot(log2(arange(len(fluct))+1),signal.medfilt(log2(fluct),8*d-1))


# In[ ]:

plot(log2(bincount(amap(lambda x: int(log2(x)), fluct[len(fluct)/2:]))))


# In[ ]:

#figure(figsize(10,10))
imshow(A)


# In[ ]:

figure(figsize=(7,7))
z=psd((sig[d:]**2)-mean(sig[d:]**2),NFFT=d, noverlap=d/2)


# In[ ]:

figure(figsize=(7,7))
plot(log2(z[1]),log2(z[0]))


# In[ ]:

from scipy import stats
print stats.kurtosis(vstack([deltas,fluct,sig,pwr, pwr**2]), axis = 1, bias=False)


# In[ ]:

print std(vstack([deltas,fluct,sig,pwr, pwr**2])/double(A.size), axis = 1)


# In[ ]:

print 2*median(pwr)/double(A.size)


# In[ ]:

figimage


# In[ ]:

subplots()[0]


# In[ ]:

gcf().figimage


# In[ ]:




# In[ ]:




# In[ ]:

x=figimage(A);


# In[ ]:

x


# In[ ]:

y=imshow(A)


# In[ ]:

y


# In[ ]:
