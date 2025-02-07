import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import apply_affine_transform
from tensorflow.image import flip_left_right, flip_up_down
from PIL import Image
import re
import os


def extract_patient_id_and_slice_number(filename):
    # Extract patient ID using regular expression
    #18_change
    #patient_match = re.search(r'MR_T1_Patient_(\d+)_defaced_slice_(\d+)\.png', filename)   #MR_T1_Patient_9083_defaced_slice_9.png
    #patient_match = re.search(r'MR_T2_GD_Patient_(\d+)_defaced_slice_(\d+)\.png', filename)
    #patient_match = re.search(r'Patient_(\d+)_images_slice_(\d+)\.png', filename)  #Patient_6016_images_slice_19.png
    patient_match = re.search(r'MR_T1_GD_Patient_(\d+)_defaced_slice_(\d+)\.png', filename) #MR_T1_GD_Patient_9083_defaced_slice_94.png
    #print(filename)
    if patient_match:
        patient_id = patient_match.group(1)
        slice_number = patient_match.group(2)
        return patient_id, slice_number
    else:
        raise ValueError(f"Could not extract patient ID and slice number from filename: {filename}")
    
def create_seg_filename(patient_id, slice_number, filename_prefix):
 #return f'{filename_prefix}Patient_{patient_id}_segslice{slice_number}.png'
    return f'{filename_prefix}Patient_{patient_id}_seg_slice_{slice_number}.png'

def load_img(filename, Nclasses, norm=False):
    ''' Load one image and its true segmentation
    '''
    patient_id, slice_number=extract_patient_id_and_slice_number(filename)
    filename_prefix = '/'.join(filename.split('/')[:-1]) + '/'
    #print(filename_prefix)
    seg_filename = create_seg_filename(patient_id, slice_number, filename_prefix)
    #T2gd_filename=f'{filename_prefix}MR_T2_GD_Patient_{patient_id}_defaced_slice_{slice_number}.png'
    #T2_filename=f'{filename_prefix}MR_T2_Patient_{patient_id}_defaced_slice_{slice_number}.png'
    CT_filename=f'{filename_prefix}CT_Patient_{patient_id}_defaced_slice_{slice_number}.png'
  
    if os.path.exists(filename):
        im_frame = Image.open(filename)
        np_frame2 = np.array(im_frame)
    else:
        np_frame2  = np.zeros_like(np_frame2)

    if os.path.exists(CT_filename):
        im_frame = Image.open(CT_filename)
        np_frame1 = np.array(im_frame)
    else:
        np_frame1  = np.zeros_like(np_frame2)

 

    im_frame = Image.open(seg_filename)
    np_frame5 = np.array(im_frame)
    
    np_frame1.astype('float32')
   
    np_frame5 = np_frame5.astype('int')
    
    np_frame1 = np_frame1[:,:,np.newaxis]
    np_frame5 = np_frame5[:,:,np.newaxis]
    
    images = np.concatenate((np_frame1,np_frame5), axis=2)
    l=len(images)
    #print(l)
    X = images[:, :, 0:1] 
    #print(X.shape)
    
    # perform normalization
    if norm == True:
        X = norm_brain(X)
    
    y = images[:, :, 1]

    #y = y / 10000  # only for brats
    #y[y==4]=3 # only for brats
    
    #y[y != 0] = 1

    #number1=0.003921568859368563
    #number2=0.007843137718737125
    #number3= 0.0117647061124444

    #y[y==number1]=1
    #y[y==number2]=2
    #y[y==number3]=3
    #print(np.sum(y == 1))
    #print(np.sum(y == 0))
    #print(np.max(y))
    
    #if Nclasses==2:
     #   y[y>0] = 1
    
    return X, y

def visualize(img, seg):
    ''' Plot the 4 differen MRI modalities and its segmentation
    '''
    if len(seg.shape)>2:
        seg = np.argmax(seg, axis=-1)
    
    #mri_names = ['T1', 'T2','T2ce', 'T1ce', 'FLAIR']
    #mri_names = ['T1ce','CT']
    mri_names = ['T1gd']
    plt.figure(figsize=(15,15))
    for i in range(1):
        plt.subplot(151)
        im = img[:,:,i]
        epsilon = 1e-10  # A small value to prevent division by zero
        im = (im - np.amin(im)) / (np.amax(im) - np.amin(im) + epsilon)
    
        
        plt.imshow(im, cmap='gray')
        plt.title(mri_names[i])
        plt.axis('off')
    seg=to_categorical(seg,4)
    #print(seg.shape)
    for i in [1,2,3,4]:
        plt.subplot(151+i)
        plt.imshow(seg[:,:,i-1], cmap='gray', vmin=0, vmax=3)
        plt.title('Seg')
        plt.axis('off')
    
    plt.show()

def visualize_save(img, seg,save_path):
    ''' Plot the 4 differen MRI modalities and its segmentation
    '''
    if len(seg.shape)>2:
        seg = np.argmax(seg, axis=-1)
    
    #mri_names = ['T1', 'T2','T2ce', 'T1ce', 'FLAIR']
    #mri_names = ['T1ce','CT']
    #19_change
    mri_names = ['CT']
    plt.figure(figsize=(15,15))
    for i in range(1):
        plt.subplot(121+i)
        im = img[:,:,i]
        epsilon = 1e-10  # A small value to prevent division by zero
        im = (im - np.amin(im)) / (np.amax(im) - np.amin(im) + epsilon)
    
        
        plt.imshow(im, cmap='gray')
        plt.title(mri_names[i])
        plt.axis('off')
    #seg=to_categorical(seg,4)
    #print(seg.shape)
    #for i in [1,2,3,4]:
     #   plt.subplot(151+i)
      #  plt.imshow(seg[:,:,i-1], cmap='gray', vmin=0, vmax=3)
      #  plt.title('Seg')
       # plt.axis('off')
    plt.subplot(122)
    plt.imshow(seg, cmap='gray', vmin=0, vmax=3)
    plt.title('Seg')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure to free memory
    
def plot_trends(scores):
    
    # Loss trend
    plt.figure(figsize=(15,20))
    plt.subplot(311)
    plt.plot(scores.history['loss'])
    plt.plot(scores.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='upper right')
    
    # Dice trend
    plt.subplot(312)
    plt.plot(scores.history['dice'])
    plt.plot(scores.history['val_dice'])
    plt.ylabel('Dice')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='lower right')
    
    #Accuracy trend
    plt.subplot(313)
    plt.plot(scores.history['accuracy'])
    plt.plot(scores.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='lower right')
    
    plt.show()
    


def plot_trends_save(scores, filename):
    # Loss trend
    plt.figure(figsize=(15,20))
    plt.subplot(311)
    plt.plot(scores.history['loss'])
    plt.plot(scores.history['val_loss'])
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='upper right')
    
    # Dice trend
    plt.subplot(312)
    plt.plot(scores.history['dice'])
    plt.plot(scores.history['val_dice'])
    plt.ylabel('Dice')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='lower right')
    
    # Accuracy trend
    plt.subplot(313)
    plt.plot(scores.history['accuracy'])
    plt.plot(scores.history['val_accuracy'])
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['train', 'valid'], loc='lower right')
    
    # Save the figure
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()  # Close the figure after saving to free memory

# Example usage:
# plot_trends(your_scores_object, 'path_to_save/your_filename.png')

import numpy as np

def apply_brightness(X, factor):
    ''' Apply brightness adjustment to the image X. '''
    # Ensure the brightness factor does not lead to overflow
    max_val = np.max(X)
    safe_factor = min(factor, np.iinfo(X.dtype).max / max_val) if max_val > 0 else factor
    return np.clip(X * safe_factor, 0, np.iinfo(X.dtype).max)

def augmentation1(X, y, do):
    ''' Apply image augmentation (rotation, translation, shear, zoom, flip, and brightness) on the image X and its segmentation y. '''
    
    # Use at least 25% original images
    if np.random.random_sample() > 0.75:
        return X, y
    
    # Affine transformation default parameters
    alpha, Tx, Ty, beta, Zx, Zy = 0, 0, 0, 0, 1, 1
    brightness_factor = 1.0

    # Rotation
    if do[0] == 1:
        alpha = (np.random.random_sample() - 0.5) * 20  # +/-10 degrees

    # Translation
    if do[1] == 1:
        Tx = np.random.uniform(-0.05 * X.shape[0], 0.05 * X.shape[0])
        Ty = np.random.uniform(-0.05 * X.shape[1], 0.05 * X.shape[1])

    # Shear
    if do[2] == 1:
        beta = (np.random.random_sample() - 0.5) * 10  # +/-5 degrees

    # Zoom
    if do[3] == 1:
        Zx, Zy = np.random.uniform(0.9, 1.1), np.random.uniform(0.9, 1.1)
    
    # Brightness
    if do[5] == 1:
        brightness_factor = np.random.uniform(1.0, 1.5)  # Up to 50% brighter
    
    X_new, y_new = apply_affine_transform(X, theta=alpha, tx=Tx, ty=Ty,
                                          shear=beta, zx=Zx, zy=Zy,
                                          row_axis=0, col_axis=1, channel_axis=2,
                                          fill_mode='constant', cval=0.0, order=1)
    y_new = apply_affine_transform(y, theta=alpha, tx=Tx, ty=Ty,
                                   shear=beta, zx=Zx, zy=Zy,
                                   row_axis=0, col_axis=1, channel_axis=2,
                                   fill_mode='constant', cval=0.0, order=0)
    
    # Apply brightness change
    X_new = apply_brightness(X_new, brightness_factor)
    X_new, y_new = X_new.numpy(), y_new.numpy()

    return X_new, y_new



def augmentation(X, y, do):
    ''' Apply image augmentation (rotation, translation, shear, zoom, and flip) on the image X and its segmentation y.
    '''
    """
    params_dir = '/local/data1/bitta693/Final_output/Output_unet1to4/'
    os.makedirs(params_dir, exist_ok=True)  # Ensure the directory exists
    
    # Filename for the parameters
    params_filename = f'params_aug.txt'
    params_filepath = os.path.join(params_dir, params_filename)
    
    # Define parameters dictionary
   
    params = {
        'rotation': alpha,
        'translation': (Tx, Ty),
        'shear': beta,
        'zoom': (Zx, Zy),
        'flipping': 'left-right' if choice == 0 else 'up-down' if choice == 1 else 'both' if choice == 2 else 'none'
    }
    # Save parameters to a file
        with open(params_filepath, 'w') as f:
        for key, value in params.items():
        f.write(f'{key}: {value}\n')
    """
    
    # lets garantee to use at least 25% of original images
    if np.random.random_sample()>0.75:
        return X, y
    
    else:
        # affine transformation default parameters
        alpha, Tx, Ty, beta, Zx, Zy = 0, 0, 0, 0, 1, 1

        ########## YOUR CODE ############
        # implement ROTATION with random angle alpha between [-45, +45) degrees
        if do[0] == 1:   
            alpha = (np.random.random_sample()-0.5)*90

        # implement TRANSLATION with random values Tx, Ty between 0 and the 10% of the image shape
        if do[1] == 1: 
            Tx, Ty = np.random.randint(X.shape[0]//10), np.random.randint(X.shape[1]//10)

        # implement SHEAR with random angle beta between [-10, 10) degrees
        if do[2] == 1: 
            beta = (np.random.random_sample()-0.5)*20

        # implement random ZOOM Zx, Zy between -20%, +20% along the 2 axis
        if do[3] == 1: 
            Zx, Zy = (1.2-0.8)*np.random.random_sample(2)+0.8
            
        ##########    END    ############

        # apply affine tranformation to the image
        X_new = apply_affine_transform(X, 
                                      theta=alpha,      # rotation
                                      tx=Tx, ty=Ty,     # translation
                                      shear=beta,       # shear
                                      zx=Zx, zy=Zy,     # zoom
                                      row_axis=0, col_axis=1, channel_axis=2, 
                                      fill_mode='constant', cval=0.0, 
                                      order=1)

        # apply affine tranformation to the target
        y_new = apply_affine_transform(y, 
                                      theta=alpha,      # rotation
                                      tx=Tx, ty=Ty,     # translation
                                      shear=beta,       # shear
                                      zx=Zx, zy=Zy,     # zoom
                                      row_axis=0, col_axis=1, channel_axis=2, 
                                      fill_mode='constant', cval=0.0, 
                                      order=0)

        # FLIPPING
        if do[4] == 1:
            choice = np.random.randint(3)

            # left-right flipping
            if choice == 0:
                X_new, y_new = flip_left_right(X_new), flip_left_right(y_new)

            # up-down flipping    
            if choice == 1:
                X_new, y_new = flip_up_down(X_new), flip_up_down(y_new)

            # both flipping
            if choice == 2:
                X_new, y_new = flip_left_right(X_new), flip_left_right(y_new)
                X_new, y_new = flip_up_down(X_new), flip_up_down(y_new)

            X_new, y_new = X_new.numpy(), y_new.numpy()

        #aug=[alpha, Tx, Ty, beta, Zx, Zy]
        #print(aug)

        return X_new, y_new



def aug_batch(Xb, yb):
    ''' Generate a augmented image batch 
    '''
    batch_size = len(Xb)
    Xb_new, yb_new = np.empty_like(Xb), np.empty_like(yb)
    
    for i in range(batch_size):
        decisions = np.random.randint(2, size=5) # 5 is number of augmentation techniques to combine
        X_aug, y_aug = augmentation(Xb[i], yb[i], decisions)
        Xb_new[i], yb_new[i] = X_aug, y_aug
        
    return Xb_new, yb_new      
    
class DataGenerator(tf.keras.utils.Sequence):
    ''' Keras Data Generator
    '''
    
    def __init__(self, list_IDs, n_classes, batch_size,n_channels, dim=(256,256),  norm=False, augmentation=False):
        'Initialization'
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.norm = norm
        self.augmentation = augmentation
        self.on_epoch_end()

    def __len__(self):
        ''' Denotes the number of batches per epoch
        '''
        return len(self.list_IDs)//self.batch_size

    def __getitem__(self, index):
        ''' Generate one batch of data
        '''
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #print(list_IDs_temp)
        # Generate data     
        X, y = self.__data_generation(list_IDs_temp)
        if self.augmentation == True:
            X, y = self.__data_augmentation(X, y)
        
        if index == self.__len__()-1:
            self.on_epoch_end()
        
        return X, y

    def on_epoch_end(self):
        ''' Updates indexes after each epoch
        '''
        self.indexes = np.arange(len(self.list_IDs))
        np.random.shuffle(self.indexes)
  
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, IDs in enumerate(list_IDs_temp):
            #print(i)
            #print(IDs)
            # Store sample
            X[i], y[i] = load_img(IDs, self.n_classes, self.norm)
            
        return X.astype('float32'), to_categorical(y, self.n_classes)
        

    def __data_augmentation(self, X, y):
        'Apply augmentation'
        X_aug, y_aug = aug_batch(X, y)
        
        return X_aug, y_aug
    

def dice_score_volume(y_true, y_pred, epsilon=1e-6):
    """
    Calculate the Dice score for binary class data.

    Args:
    - y_true: array, ground truth
    - y_pred: array, predictions
    - epsilon: small constant to avoid division by zero

    Returns:
    - dice: Dice score
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection + epsilon) / (np.sum(y_true) + np.sum(y_pred) + epsilon)

