from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Activation, BatchNormalization, MaxPool2D, Conv2DTranspose, Dropout, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import tensorflow as tf 
class unet(object):
    #16_change
    def __init__(self, img_size, Nclasses, class_weights, weights_name='myWeights.h5', Nfilter_start=64, depth=4):
        self.img_size = img_size
        self.Nclasses = Nclasses
        self.class_weights = class_weights
        self.weights_name = weights_name
        self.Nfilter_start = Nfilter_start
        self.depth = depth

        inputs = Input(img_size)
        
        def dice(y_true, y_pred, w=self.class_weights):
            '''
            y_true = tf.convert_to_tensor(y_true, 'float32')
            y_pred = tf.convert_to_tensor(y_pred, 'float32')

            num = 2 * tf.reduce_sum(tf.reduce_sum(y_true*y_pred, axis=[0,1,2])*w)
            den = tf.reduce_sum(tf.reduce_sum(y_true+y_pred, axis=[0,1,2])*w) + 1e-5
            '''
            w = tf.constant(w, dtype=tf.float32)  # Example if w needs to be used and was a numpy array
    
            # Use TensorFlow operations for calculations
            num = 2 * tf.reduce_sum(y_true * y_pred, axis=[0, 1, 2]) * w
            den= tf.reduce_sum(y_true + y_pred, axis=[0, 1, 2]) * w+ 1e-5
            return num/den
    
        def diceLoss(y_true, y_pred):
            return 1-dice(y_true, y_pred)          
        




        
        ########## YOUR CODE ############
        
        # This is a help function that performs 2 convolutions, each followed by batch normalization
        # and ReLu activations, Nf is the number of filters, filter size (3 x 3)
        def convs(layer, Nf):
            x = Conv2D(Nf, (3,3), kernel_initializer='he_normal', padding='same')(layer)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            x = Conv2D(Nf, (3,3), kernel_initializer='he_normal', padding='same')(x)
            x = BatchNormalization()(x)
            x = Activation('relu')(x)
            return x
            
        # This is a help function that defines what happens in each layer of the encoder (downstream),
        # which calls "convs" and then Maxpooling (2 x 2). Save each layer for later concatenation in the upstream.
        def encoder_step(layer, Nf):
            y = convs(layer, Nf)
            x = MaxPool2D(pool_size=(2,2))(y)
            return y, x
            
        # This is a help function that defines what happens in each layer of the decoder (upstream),
        # which contains upsampling (2 x 2), 2D convolution (2 x 2), batch normalization, concatenation with 
        # corresponding layer (y) from encoder, and lastly "convs"
        def decoder_step(layer, layer_to_concatenate, Nf):
            x = Conv2DTranspose(filters=Nf, kernel_size=(3,3), strides=(2,2), padding='same', kernel_initializer='he_normal')(layer)
            x = BatchNormalization()(x)
            x = concatenate([x, layer_to_concatenate])
            x = convs(x, Nf)
            return x
            
        layers_to_concatenate = []
        x = inputs
        
        # Make encoder with 'self.depth' layers, 
        # note that the number of filters in each layer will double compared to the previous "step" in the encoder
        for d in range(self.depth-1):
            y,x = encoder_step(x, self.Nfilter_start*np.power(2,d))
            layers_to_concatenate.append(y)
            
        # Make bridge, that connects encoder and decoder using "convs" between them. 
        # Use Dropout before and after the bridge, for regularization. Use drop probability of 0.2.
        #17_change
        x = Dropout(0.2)(x)
        x = convs(x,self.Nfilter_start*np.power(2,self.depth-1))
        x = Dropout(0.2)(x)        
        
        # Make decoder with 'self.depth' layers, 
        # note that the number of filters in each layer will be halved compared to the previous "step" in the decoder
        for d in range(self.depth-2, -1, -1):
            y = layers_to_concatenate.pop()
            x = decoder_step(x, y, self.Nfilter_start*np.power(2,d))            
            
        ##########    END    ############
        
        # Make classification (segmentation) of each pixel, using convolution with 1 x 1 filter
        final = Conv2D(filters=self.Nclasses, kernel_size=(1,1), activation = 'softmax', padding='same', kernel_initializer='he_normal')(x)
        
        # Create model
        self.model = Model(inputs=inputs, outputs=final)
        self.model.compile(loss=diceLoss, optimizer=Adam(learning_rate=5e-5), metrics=['accuracy',dice])
        
    def train(self, train_gen, valid_gen, nEpochs):
        print('Training process:')       
        callbacks = [ModelCheckpoint(self.weights_name, save_best_only=True, save_weights_only=True),
                     EarlyStopping(patience=40)]
        
        history = self.model.fit(train_gen, validation_data=valid_gen, epochs=nEpochs, callbacks=callbacks)

        return history    
    
    def evaluate(self, test_gen):
        print('Evaluation process:')
        score, acc, dice = self.model.evaluate(test_gen)
        print('Accuracy: {:.4f}'.format(acc*100))
        print('Dice: {:.4f}'.format(dice*100))
        return acc, dice
    
    def predict(self, X):
        y_pred = self.model.predict(X)
        return y_pred

    def calculate_metrics(self, y_true_flat, y_pred_flat):
        ########## YOUR CODE ############
        # be sure that the inputs are binary.
        # calculate the confusion matrix using tf.math.confusion_matrix()
        
        cm = tf.math.confusion_matrix(y_true_flat, y_pred_flat, num_classes=2).numpy() #confusion_matrix(y_true_flat, y_pred_flat, labels=[0,1]) 
        
        # calculate the accuracy and Dice using. 
        # Set to np.nan the value for Dice when the confusion matrix has only true negatives.
        
        acc = np.trace(cm)/(np.sum(cm))
        #if cm[0,0] == 256*256:
         #   dice = 0
        #else:
        dice = 2*cm[1,1]/(2*cm[1,1]+cm[1,0]+cm[0,1])
        
        return acc, dice
    
        ##########    END    ############
    
    def get_metrics(self, generator):
        ''' This function calculates the metrics accuracy and Dice for each image contained in the input generator.
        '''
        Nim = len(generator)*generator.batch_size
        ACC = np.empty((Nim, self.Nclasses))
        DICE = np.empty((Nim, self.Nclasses))
        n = 0
        for i in range(len(generator)):
            X_batch, y_batch = generator[i]
            y_pred = self.model.predict(X_batch)
            y_pred = to_categorical(tf.argmax(y_pred, axis=-1), self.Nclasses)

            for c in range(Nclasses):
                y_true_flat = tf.reshape(y_batch[0,:,:,c], (256*256,))
                y_pred_flat = tf.reshape(y_pred[0,:,:,c], (256*256,))            

                acc, dice = self.calculate_metrics(y_true_flat, y_pred_flat)
                ACC[n,c] = acc
                DICE[n,c] = dice

            n+=1

        return ACC, DICE
    
    
    

    def dice_per_class(self, y_true, y_pred, epsilon=1e-5):
        # Assuming y_true and y_pred are one-hot encoded, with shape (batch_size, height, width, Nclasses)
        # Adjust the axes to match your data if necessary
        axes = (0,1,2)  # Skip the batch axis for summation
        intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
        union = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)
        dice_scores = (2. * intersection + epsilon) / (union + epsilon)
        return dice_scores  # Shape: (Nclasses,)

    
    def evaluate_per_class(self, test_gen):
        print('Evaluation process:')
        y_pred = []
        y_true = []
        
        for x_batch, y_batch in test_gen:
            y_pred_batch = self.model.predict(x_batch)
            y_pred.append(y_pred_batch)
            y_true.append(y_batch)
        y_pred = np.concatenate(y_pred, axis=0)
        y_true = np.concatenate(y_true, axis=0)
        # Convert y_pred to one-hot format if necessary
        # y_pred = tf.one_hot(tf.argmax(y_pred, axis=-1), depth=self.Nclasses)
        dice_scores = self.dice_per_class(y_true, y_pred) 
        for i, score in enumerate(dice_scores):
            print(f"Class {i} Dice Score: {score:.4f}")
        return dice_scores
