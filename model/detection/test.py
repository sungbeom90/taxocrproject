import tensorflow.keras.backend as K

x = K.constant(x)
x = np.reshape(x, ((1,) + x.shape))
y = np.reshape(y, ((1,) + y.shape))
y = np.float32(y)
y = K.constant(y)

demo.compile(optimizer='adam',
              loss=loss_region,
              metrics=['accuracy'])

print(x.shape)
print(y.shape)
demo.fit(x,
          y,
          batch_size=1, 
          epochs=1, verbose=True)