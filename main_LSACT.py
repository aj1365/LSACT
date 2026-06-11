import numpy as np
from utils import *
from tensorflow.keras.models import clone_model
import matplotlib.pyplot as plt

dataset = "UC"  # UC, AID, NWPU, EuroSAT_RGB

SEED = 0
train_ds, train_labels, val_ds, val_labels, test_ds, test_labels, class_names = get_dataset(dataset = dataset, seed = SEED)
NUM_CLASSES = len(class_names)

for idx, cls in enumerate(class_names):
    n_train = train_labels.count(idx)
    n_val   = val_labels.count(idx)
    n_test  = test_labels.count(idx)
    total   = n_train + n_val + n_test
    print(f"{cls:25s}  Train: {n_train:3d}   Val: {n_val:3d}   Test: {n_test:3d}   Total: {total:3d}")
#print(f"\nTrain: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")

from model_LSACT import LSACT_Tiny
model =   LSACT_Tiny(input_shape= None, num_classes=NUM_CLASSES, target_size = 64)
model.summary()



import keras
checkpoint = keras.callbacks.ModelCheckpoint(
            "Best_Model_Weights.h5",
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
# Define a callback to modify the learning rate dynamically
lr_callback = keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5,
            patience=10,
            min_lr=1e-5
            )
        
history = model.fit(train_ds, 
                        validation_data=val_ds, 
                        epochs=50, 
                        callbacks=[checkpoint, lr_callback]
                        )

display_history(history)
    
    
model.load_weights("Best_Model_Weights.h5")
    
oa, aa, kappa, cm, each_acc, f1 = evaluate_model(model, test_ds, class_names, verbose = 1)
    #each_acc = [round(item, 2) for item in each_acc*100]
    
f1 = [round(item, 2) for item in f1]

   
