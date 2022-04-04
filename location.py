import tensorflow as tf
from tensorflow.keras.layers import Dense,Flatten,MaxPool2D,Conv2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16
train=ImageDataGenerator(rescale=1/255.)
valid=ImageDataGenerator(rescale=1/255.)

train=train.flow_from_directory('car-damage-dataset/data2a/training',
                                 shuffle=True,batch_size=16,
                                  
                                class_mode='categorical', target_size=(128, 128))
valid=valid.flow_from_directory('car-damage-dataset/data2a/validation',
                                  shuffle=False,batch_size=16,
                                class_mode='categorical', target_size=(128, 128))

modelv=VGG16(include_top=False,  classes=3, classifier_activation='softmax')
modelv.trainable=False
inputs=tf.keras.layers.Input(shape=(128,128,3))
x=modelv(inputs)
x=tf.keras.layers.GlobalAveragePooling2D()(x)
outputs=Dense(3,activation='softmax')(x)
model=tf.keras.Model(inputs,outputs)

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.CategoricalCrossentropy(), metrics='accuracy')
history=model.fit(train,epochs=10,
          steps_per_epoch=len(train),
          validation_data=valid,
          validation_steps=len(valid)
          )

modelv.trainable = True

# Freeze all layers except for the
for layer in modelv.layers[:-10]:
  layer.trainable = False

# Recompile the model (always recompile after any adjustments to a model)
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(lr=0.0001), # lr is 10x lower than before for fine-tuning
              metrics=["accuracy"])
initial_epochs=10
fepochs=initial_epochs+5
history=model.fit(train,
            epochs=fepochs,
            initial_epoch=history.epoch[-1],
            steps_per_epoch=len(train),
            validation_data=valid,
            validation_steps=len(valid))

model.save('data2a.h5')