import VAE
vae = train(network_architecture, training_epochs=2)
x_sample, y_sample = mnist.test.next_batch(5)
d4cluster=vae.transform(x_sample).astype(np.float)
center,result=kmean2.TFKMeansCluster(d4cluster,2)
print (np.reshape(center,[2,2])