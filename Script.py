# for auto-reloading external modules
# see http://stackoverflow.com/questions/1907993/autoreload-of-modules-in-ipython
#GPU
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True) 

#1_change
#model description
print('Modeldescription:Inputchannel:1 CT ,Outputchannel:4, Aug:T, Epoch:300 ,Norm:N ,dense:4 Dropout:20' )

      
#Load Data
from  Util import *
import numpy as np
import matplotlib.pyplot as plt
import glob
from Model import *
import tensorflow as tf 


def load_list(file_path):
    with open(file_path, 'r') as file:
        data_list = [line.strip() for line in file]
    return data_list

list_train = load_list('train_dataset_list.txt')
list_test = load_list('test_dataset_list.txt')
list_valid = load_list('valid_dataset_list.txt')
#2_change: applied when we have normalization
##data adress
#list_train = sorted(glob.glob('/local/data1/bitta693/dataset/train_dataset_slices/*MR_T1_GD_Patient*'))
Nim_tr= len(list_train)

#3_change
#Number of classes
Nclasses = 4


print('The train image dataset contatins: {} images.'.format(Nim_tr))



#list_test= sorted(glob.glob('/local/data1/bitta693/dataset/test_dataset_slices/*MR_T1_GD_Patient*'))
Nim_te = len(list_test)


print('The test image dataset contatins: {} images.'.format(Nim_te))


#list_valid= sorted(glob.glob('/local/data1/bitta693/dataset/valid_dataset_slices/*MR_T1_GD_Patient*'))
Nim_v= len(list_valid)

           


      
train_gen = DataGenerator(list_train, n_classes=Nclasses,batch_size=8,n_channels=1,augmentation=True)
valid_gen = DataGenerator(list_valid, n_classes=Nclasses,batch_size=8,n_channels=1)
test_gen = DataGenerator(list_test, n_classes=Nclasses,batch_size=1,n_channels=1)

##########    END    ############

Ntrain = len(list_train)
Nvalid = len(list_valid)
Ntest = len(list_test)


#print('The training, validation, and testing set have {} ({:.2f}%), {} ({:.2f}%) and {} ({:.2f}%) images respectively.'
 #     .format(Ntrain, 100*Ntrain/Nim, Nvalid, 100*Nvalid/Nim, Ntest, 100*Ntest/Nim))



#save image of input and output
idx = np.random.randint(len(train_gen))
#print(idx)
# load the idx-th batch
Xbatch, ybatch = train_gen[idx]
print('A image batch has shape: {}'.format(Xbatch.shape))
print('A target batch has shape: {}'.format(ybatch.shape))

#5_change
# visualize few data samples 
for i in range(8):
    filename = f"T1GD&seg4classes_{i}.png"
    visualize_save(Xbatch[i], ybatch[i], filename)


#class weightY = []
Y = []
n = 0


for _, ybatch in train_gen:
    # concatenate in Y the n-th flattened target batch
    Y = np.concatenate((Y, np.argmax(ybatch, axis=-1).flatten()))
    
    # interrupt the cycle
    n += 1
    if n == 50:
        break

Y = Y.astype('int64')


Nsamples = len(Y)
class_weights = (Nsamples/(Nclasses*np.bincount(Y))).astype('float32')
print(Nsamples)
print(Nclasses)
print(np.bincount(Y))

print('The class weights for the four different classes are respectively:\n{}'.format(class_weights))
print(class_weights.dtype)

#6_change
#model 
img_size = Xbatch.shape[1:]
net = unet(img_size, Nclasses, class_weights, weights_name='myWeights_multiseg.h5')
net.model.summary()
from contextlib import redirect_stdout

#7_change
# Specify the filename where you want to save the model summary
summary_filename = f"model_summary.txt"

with open(summary_filename, 'w') as f:
    with redirect_stdout(f):
        net.model.summary()

#train
#8_change
results = net.train(train_gen, valid_gen, nEpochs=300)

#9_change
filename = f"trends.png"
plot_trends_save(results,filename)

      
#10_change
save_dir = "pred_GroundTruth_slices_valid"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

for i in range(len(list_test)):
    filename = list_test[i]
    filename_parts = filename.split('/')[-1].split('_')
    patient_id = filename_parts[4]
    slice_number = filename_parts[7].split('.')[0]
    #11_change
    valid_gen = DataGenerator([filename], n_classes=Nclasses, batch_size=1, n_channels=1)
    X_batch, y_batch = valid_gen[0]
    y_pred = net.predict(X_batch)
    
    # Define file paths
    y_true_path = os.path.join(save_dir, f"y_true_patient_{patient_id}_slice_{slice_number}.npy")
    y_pred_path = os.path.join(save_dir, f"y_pred_patient_{patient_id}_slice_{slice_number}.npy")
    
    # Save the ground truth and prediction
    np.save(y_true_path, y_batch)
    np.save(y_pred_path, y_pred)



#make a volume
from collections import defaultdict
import numpy as np
import os
import glob
def extract_patient_id_and_slice_number(filename):
    parts = filename.split('_')
    # Assuming 'patient' and 'slice' always precede their respective numbers
    # and these keywords are consistently placed in the filenames
    patient_index = parts.index('patient') + 1
    slice_index = parts.index('slice') + 1
    
    patient_id = parts[patient_index]
    slice_number = int(parts[slice_index].split('.')[0])  # Removing file extension and converting to int
    return patient_id, slice_number
      
#12_change
# Directory where the files are saved
volume_save_dir = "aggregated_volumes"

# Directory to save the aggregated volumes
#volume_save_dir = os.path.join(save_dir, "aggregated_volumes")
if not os.path.exists(volume_save_dir):
    os.makedirs(volume_save_dir)
print(volume_save_dir)
# Initialize dictionaries to hold file paths, sorted by patient and type
y_true_files = defaultdict(list)
y_pred_files = defaultdict(list)

# Gather .npy file paths and organize them by patient ID and type (true or pred)
for file_path in glob.glob(os.path.join(save_dir, "*.npy")):
    filename = os.path.basename(file_path)
    patient_id, slice_number = extract_patient_id_and_slice_number(filename)
    
    if "y_true" in filename:
        y_true_files[patient_id].append((slice_number, file_path))
    elif "y_pred" in filename:
        y_pred_files[patient_id].append((slice_number, file_path))

# Sort the files for each patient by slice number

for patient_id in y_true_files:
    y_true_files[patient_id].sort()
for patient_id in y_pred_files:
    y_pred_files[patient_id].sort()

# Function to load and stack slices to form a volume
def stack_slices(file_paths):
    slices = [np.load(path) for _, path in file_paths]
    volume = np.vstack(slices)
    return volume

# Create and save volumes for each patient
for patient_id in y_true_files:
    # Stack and save y_true volume
    volume_true = stack_slices(y_true_files[patient_id])
    true_volume_path = os.path.join(volume_save_dir, f"volume_true_patient_{patient_id}.npy")
    np.save(true_volume_path, volume_true)

for patient_id in y_pred_files:
    # Stack and save y_pred volume
    volume_pred = stack_slices(y_pred_files[patient_id])
    pred_volume_path = os.path.join(volume_save_dir, f"volume_pred_patient_{patient_id}.npy")
    
    np.save(pred_volume_path, volume_pred)

print("All volumes have been saved successfully.")


#dice 
import numpy as np
import re



# Assuming 'list_valid' contains paths to validation images
patient = []
for filename in list_test:
    #13_change
    patient_match = re.search(r'MR_T1_GD_Patient_(\d+)_defaced_slice_(\d+)\.png', filename)
    if patient_match:
        patient_id = patient_match.group(1)
        patient.append(patient_id)
patient = np.unique(patient)



from collections import defaultdict
import numpy as np
import os
import glob

def extract_patient_id_and_slice_number(filename):
    parts = filename.split('_')
    # Assuming 'patient' and 'slice' always precede their respective numbers
    # and these keywords are consistently placed in the filenames
    patient_index = parts.index('patient') + 1
    slice_index = parts.index('slice') + 1
    
    patient_id = parts[patient_index]
    slice_number = int(parts[slice_index].split('.')[0])  # Removing file extension and converting to int
    return patient_id, slice_number


# Function to load and stack slices to form a volume
def stack_slices(file_paths):
    slices = [np.load(path) for _, path in file_paths]
    volume = np.vstack(slices)
    return volume




#14_change

# Specify the output file where Dice scores will be saved
output_file_path = 'dice_scores.txt'
dice_scores_matrix = np.zeros((len(patient), 4))
l=0
# Open the output file in append mode
with open(output_file_path, 'a') as output_file:
    for patient_id in patient:
        #15_change
        volume_true_path = f"/local/data1/bitta693/python_code/utils_code/home_folder/CT_test4__nonzero/aggregated_volumes/volume_true_patient_{patient_id}.npy"
        volume_predict_path = f"/local/data1/bitta693/python_code/utils_code/home_folder/CT_test4__nonzero/aggregated_volumes/volume_pred_patient_{patient_id}.npy"
        print(volume_true_path)
        print(volume_predict_path)
        volume_true = np.load(volume_true_path)
        volume_predict = np.load(volume_predict_path)

        dice_scores = []
        for i in range(4):  # Loop over each class
            y_true_class = volume_true[:, :, :, i]
            y_pred_class = volume_predict[:, :, :, i]

            y_true_binary = (y_true_class > 0.5).astype(int)
            y_pred_binary = (y_pred_class > 0.5).astype(int)

            score = dice_score_volume(y_true_binary, y_pred_binary)
            dice_scores.append(score)
            output_file.write(f"Patient ID {patient_id}, Dice score for class {i}: {score}\n")
        dice_scores_matrix[l]= dice_scores
        l=l+1
        overall_dice_score = np.mean(dice_scores[1:4])
        output_file.write(f"Patient ID {patient_id}, Overall Dice score: {overall_dice_score}\n")
    mean_dice_scores = np.mean(dice_scores_matrix[:, 1:4], axis=0)
    print(mean_dice_scores)
# After running this, check the file '/local/data1/bitta693/dataset/dice_scores.txt' for the output

print('complete')
