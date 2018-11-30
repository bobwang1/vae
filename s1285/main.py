import VAE_vMF
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import kmean2
network_architecture = \
   dict(n_hidden_recog_1=500, # 1st layer encoder neurons
        n_hidden_recog_2=500, # 2nd layer encoder neurons
        n_hidden_gener_1=500, # 1st layer decoder neurons
        n_hidden_gener_2=500, # 2nd layer decoder neurons
        n_input=784, # MNIST data input (img shape: 28*28)
        n_z=20)  # dimensionality of latent space
vae = VAE_vMF.train(network_architecture, training_epochs=1)
x_sample, y_sample = VAE_vMF.mnist.test.next_batch(1000)
d4cluster=vae.transform(x_sample).astype(np.float)
#d4cluster=vae.transform(x_sample)
#center,result=kmean2.TFKMeansCluster(d4cluster,10)
result=KMeans(n_clusters=10,random_state=100).fit(x_sample).labels_

true=0
num=0
x=[]
for i in range (10):
    b=[j for j,v in enumerate(result) if v==i]
    for k in b:
        x.append((np.argwhere(y_sample[k,:]==1)))
    x=list(map(int,x))
    real= max(x,key =x.count)
    for m in b:
        result[m]=real
    x=[]
for i in range (np.size(result)):
    if y_sample[i,result[i]]==1:
        true+=1
print(true/np.size(result))
#
#plt.figure(figsize=(8, 6))
#plt.subplot(121)
#plt.scatter(d4cluster[:,0],d4cluster[:,1], c=result)
#plt.grid()
#plt.subplot(122)
#plt.scatter(d4cluster[:,0],d4cluster[:,1], c=np.argmax(y_sample, 1))
#plt.colorbar()
#plt.grid()
#plt.show()
#Tf_png='test3.png'
#plt.savefig(Tf_png,dpi=150)
# print (np.reshape(center,[-1,2]))


# print center.shape
# plt.figure(figsize=(8, 6))
# plt.scatter(center[:, 0], z_mu[:, 1], c=result)
# plt.colorbar()
# plt.grid()
# plt.show()
# Tf_png='test.png'
# plt.savefig(Tf_png,dpi=150)

# tneinput=vae.transform(x_sample)


# x_sample = mnist.test.next_batch(100)[0]
x_reconstruct = vae.reconstruct(x_sample)


plt.figure(figsize=(8, 12))
for i in range(5):
    plt.subplot(5, 2, 2*i + 1)
    plt.imshow(x_sample[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Test input")
    plt.colorbar()
    plt.subplot(5, 2, 2*i + 2)
    plt.imshow(x_reconstruct[i].reshape(28, 28), vmin=0, vmax=1, cmap="gray")
    plt.title("Reconstruction")
    plt.colorbar()
plt.tight_layout()
Re_png='Reconstructionvmf.png'
plt.savefig(Re_png,dpi=150)

#x_sample, y_sample = mnist.test.next_batch(5000)
#network_architecture = \
#   dict(n_hidden_recog_1=500, # 1st layer encoder neurons
#        n_hidden_recog_2=500, # 2nd layer encoder neurons
#        n_hidden_gener_1=500, # 1st layer decoder neurons
#        n_hidden_gener_2=500, # 2nd layer decoder neurons
#        n_input=784, # MNIST data input (img shape: 28*28)
#        n_z=2)  # dimensionality of latent space
#
#vae_2d = train(network_architecture, training_epochs=5)
#z_mu = vae_2d.transform(x_sample)


# plt.figure(figsize=(8, 6))
# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1))
# plt.colorbar()
# plt.grid()
# plt.show()
# Tf_png='test.png'
# plt.savefig(Tf_png,dpi=150)



# X_tsne = TSNE(n_components=2,random_state=33).fit_transform(tneinput)
#
# plt.figure(figsize=(10, 5))
# plt.subplot(121)
# plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=np.argmax(y_sample, 1),label="t-SNE")
# plt.legend()
# plt.subplot(122)
# plt.scatter(z_mu[:, 0], z_mu[:, 1], c=np.argmax(y_sample, 1),label="old")
# plt.legend()
# plt.show()
#
# nx = ny = 20
# x_values = np.linspace(-3, 3, nx)
# y_values = np.linspace(-3, 3, ny)
#
# canvas = np.empty((28*ny, 28*nx))
# for i, yi in enumerate(x_values):
#   for j, xi in enumerate(y_values):
#        z_mu = np.array([[xi, yi]]*vae.batch_size)
#        x_mean = vae_2d.generate(z_mu)
#        canvas[(nx-i-1)*28:(nx-i)*28, j*28:(j+1)*28] = x_mean[0].reshape(28, 28)
#
#plt.figure(figsize=(8, 10))
#Xi, Yi = np.meshgrid(x_values, y_values)
#plt.imshow(canvas, origin="upper",cmap="gray")
#plt.tight_layout()