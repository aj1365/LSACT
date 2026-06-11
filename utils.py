import numpy as np
import matplotlib.pyplot as plt
from operator import truediv
import os, glob, random
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, cohen_kappa_score, accuracy_score, f1_score
from tqdm import tqdm






def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc



def display_history(history):
    # Retrieve loss and accuracy data
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(loss) + 1)
    
    # Create a figure with 2 horizontal subplots
    plt.figure(figsize=(12, 5))
    
    # Subplot for training and validation loss
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, first subplot
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Subplot for training and validation accuracy
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, second subplot
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    # Show the combined figure
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
    
    # Get training history
    val_acc = history.history['val_accuracy']
    best_epoch = np.argmax(val_acc)
    best_val = val_acc[best_epoch]

    
    # Plot Accuracy
    plt.figure(figsize=(10, 4))
    plt.plot(history.history['accuracy'], 'y', label='Train Acc')
    plt.plot(val_acc, 'r', label='Val Acc')
    
    # Mark best epoch
    plt.axvline(best_epoch, color='k', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.scatter(best_epoch, best_val, color='black')
    plt.text(best_epoch, best_val, f"{best_val:.2f}", fontsize=10, color='black', va='bottom')
    
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
     

def show_random_test_prediction(model, test_ds, class_names, shuffle_buf=10000):
    # flatten to individual examples, shuffle every call, then take 1
    for img, label in test_ds.unbatch().shuffle(shuffle_buf, reshuffle_each_iteration=True).take(1):
        img = img.numpy()
        # handle one-hot or integer labels
        if tf.rank(label) == 0:
            true_id = int(label.numpy())
        else:
            true_id = int(tf.argmax(label).numpy())
        true_name = class_names[true_id]

        # predict
        logits = model.predict(img[None, ...], verbose=0)
        pred_id = int(np.argmax(logits[0]))
        pred_name = class_names[pred_id]
        conf = float(np.max(logits[0]))

        # visualize
        plt.figure(figsize=(5,5))
        plt.imshow(img)
        plt.axis("off")
        color = "green" if pred_id == true_id else "red"
        plt.title(f"Pred: {pred_name} (p={conf:.3f})\nTrue: {true_name}", color=color, fontsize=12)
        plt.show()



def evaluate_model(model, test_ds, class_names, verbose = 0):
    # Collect predictions and true labels
    y_true, y_pred = [], []
    for batch_x, batch_y in test_ds:
        probs = model.predict(batch_x, verbose = 1)       # (batch, num_classes)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(np.argmax(batch_y.numpy(), axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall Accuracy
    oa = accuracy_score(y_true, y_pred)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=range(len(class_names)))
    annot_labels = np.where(
    cm > 0,
    np.round(cm, 2).astype(str),   # convert to strings with 2 decimals
    ""                              # hide zeros
)
    if verbose == 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=annot_labels, fmt='', cmap='Blues',
                    xticklabels=class_names,
                    yticklabels=class_names,
                    annot_kws={"size": 6, "weight": "bold"},
                    cbar=False)
        plt.gca().set_aspect('equal', adjustable='box')  # keep cells square
        plt.xlabel('Predicted')
        plt.xticks(rotation=45, ha='right') # 'ha' for horizontal alignment
        plt.ylabel('True')
        #plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.show()
    # Average Accuracy (per-class accuracies → mean)
    each_acc, aa = AA_andEachClassAccuracy(cm)
    #class_acc = cm.diagonal() / cm.sum(axis=1)
    #aa = np.mean(class_acc)

    # Cohen’s Kappa
    kappa = cohen_kappa_score(y_true, y_pred)
    
    f1  = f1_score(y_true, y_pred, average=None)

    # Print
    print(f"✅ Overall Accuracy (OA): {oa*100:.2f}%")
    print(f"📊 Average Accuracy (AA): {aa*100:.2f}%")
    print(f"📈 Kappa Score: {kappa*100:.2f}")

    return oa, aa, kappa, cm, each_acc, f1



def get_dataset(dataset, seed):
        
    if dataset == "UC":
        DATA_DIR = "UCMerced_LandUse"
        RAW_DECODE_SIZE = (256, 256)
        
    elif dataset == "AID":
        DATA_DIR = "AID"
        RAW_DECODE_SIZE = (600, 600)
        
    elif dataset == "NWPU":
        DATA_DIR = "NWPU-RESISC45"
        RAW_DECODE_SIZE = (256, 256)
             
    else:
        DATA_DIR = 'EuroSAT_RGB'
        RAW_DECODE_SIZE = (64, 64)
    
    
    BATCH    = 8
    SEED     = seed #1337
    AUTOTUNE = tf.data.AUTOTUNE
    
    tr_per = 0.7
       
    
    # ---------- enumerate files per class ----------
    class_names = sorted([d for d in os.listdir(DATA_DIR)
                          if os.path.isdir(os.path.join(DATA_DIR, d))])
    
    def list_images(cls):
        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        paths = []
        for e in exts:
            paths.extend(glob.glob(os.path.join(DATA_DIR, cls, e)))
        return sorted(paths)
    
    per_class = {c: list_images(c) for c in class_names}
    NUM_CLASSES = len(class_names)
    
    # ---------- stratified split 70/15/15, no overlap ----------
    train_files, val_files, test_files = [], [], []
    train_labels, val_labels, test_labels = [], [], []
    
    for label, cls in enumerate(class_names):
        val_per = random.randint(13,15)/100
        files = per_class[cls]
        # deterministic per-class shuffle
        rnd = random.Random(SEED)
        rnd.shuffle(files)
    
        n = len(files)
        n_train = int(round(tr_per * n))
        n_val   = int(round(val_per * n))
        n_test  = n - n_train - n_val
        if n_test < 0:  # rare rounding fix
            n_val += n_test
            n_test = 0
    
        tr = files[:n_train]
        va = files[n_train:n_train+n_val]
        te = files[n_train+n_val:]
    
        train_files += tr; train_labels += [label]*len(tr)
        val_files   += va; val_labels   += [label]*len(va)
        test_files  += te; test_labels  += [label]*len(te)
    
    # safety: ensure no overlap
    def _assert_disjoint(a, b, c):
        sa, sb, sc = set(a), set(b), set(c)
        assert sa.isdisjoint(sb) and sa.isdisjoint(sc) and sb.isdisjoint(sc), "Overlap detected!"
    _assert_disjoint(train_files, val_files, test_files)
    
    print(f"Classes: {NUM_CLASSES}")
    print(f"Train: {len(train_files)}  Val: {len(val_files)}  Test: {len(test_files)}")
    
    
    def _read_any_image_py(path_bytes):
        p = path_bytes.decode("utf-8")
        with Image.open(p) as im:
            im = im.convert("RGB")               # handles TIFF/PNG/JPEG/… → RGB
            arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr
    
    
    def make_ds(paths, labels, shuffle=False):
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    
        def decode(path, label):
            img = tf.numpy_function(_read_any_image_py, [path], tf.float32)
            img.set_shape([None, None, 3])       # important for shape inference
            img = tf.image.resize(img, RAW_DECODE_SIZE)
            y   = tf.one_hot(label, depth=NUM_CLASSES)
            return img, y
    
        if shuffle:
            ds = ds.shuffle(buffer_size=min(5000, len(paths)), seed=SEED, reshuffle_each_iteration=True)
    
        ds = ds.map(decode, num_parallel_calls=AUTOTUNE)
        # Strongly recommended to cache when decoding in Python:
        ds = ds.cache()                          # RAM cache; or .cache("ucm_cache_train")
        ds = ds.batch(BATCH).prefetch(AUTOTUNE)
        return ds
    
    train_ds = make_ds(train_files, train_labels, shuffle=True)
    val_ds   = make_ds(val_files,   val_labels,   shuffle=False)
    test_ds  = make_ds(test_files,  test_labels,  shuffle=False)
    
    return train_ds, train_labels, val_ds, val_labels, test_ds, test_labels, class_names
    
    
    
    
    
    
def get_model(Name, NUM_CLASSES):
    if Name == "ViT":
        from Models.ViTClassifier import ViT
        model = ViT(None, num_classes = NUM_CLASSES, patch_size=4,  target_size = 64)

    elif Name == "ResNet":
        from Models.Transfer import ResNet_152_V2
        model = ResNet_152_V2(None, num_classes = NUM_CLASSES, train_base=True,  target_size = 64)
        
    elif Name == "VGG_19":
         from Models.Transfer import VGG_19
         model = VGG_19(None, num_classes = NUM_CLASSES, train_base=True,  target_size = 64)

    elif Name == "MobileNet_V2":
        from Models.Transfer import MobileNet_V2
        model = MobileNet_V2(None, num_classes = NUM_CLASSES, train_base=True, target_size = 64)

    elif Name == "EfficientNet_B0":
        from Models.Transfer import EfficientNet_B0
        model = EfficientNet_B0(None, num_classes = NUM_CLASSES, train_base=True, target_size = 64)

    elif Name == "SceneMixer":
        from Models.SceneMixer import SceneMixer
        model =   SceneMixer(None, num_classes=NUM_CLASSES, target_size = 64)

    elif Name == "LSACT":
        from Models.LSACT11 import LSACT_Tiny
        model =   LSACT_Tiny(input_shape= None, num_classes=NUM_CLASSES, target_size = 64)

    elif Name == "MBLANet":
        from Models.MBLANet import MBLANet
        model = MBLANet(input_shape=None, num_classes=NUM_CLASSES, target_size = 64)

    elif Name == "MSHCCT":
        from Models.MSHCCT import MSHCCT
        model = MSHCCT(input_shape=None, num_classes=NUM_CLASSES, target_size = 64)
        
    elif Name == "MDRCN":
         from Models.MDRCN import MDRCN
         model = MDRCN(input_shape=None, num_classes=NUM_CLASSES, target_size = 64 )

    elif Name == "STConvNeXt":
        from Models.STConvNeXt import STConvNeXt
        model = STConvNeXt(input_shape=None, num_classes=NUM_CLASSES, target_size = 64)
    
    elif Name == "SceneFormer":
        from Models.SceneFormer import SceneFormer
        model = SceneFormer(input_shape=None, num_classes=NUM_CLASSES, target_size = 64)
        
    elif Name == "STConvNeXt1":
        from Models.STConvNeXt1 import build_and_compile_stconvnext
        model = build_and_compile_stconvnext(input_shape=None, num_classes=NUM_CLASSES)
        
    return model

    

def get_prediections(model, test_ds):
    images, labels = extract_from_dataset(test_ds)
    labels = np.argmax(labels, axis=1)
        
    preds = []
    for start in tqdm(range(0, images.shape[0], 16)):
        end = start + 16
        batch_preds = np.argmax(model.predict(images[start:end], verbose=0), axis = 1)
        preds.extend(batch_preds)
   
    return images, labels, preds
    
    

def show_predictions(images, labels, preds, class_names=None, idx = 0):
    plt.figure(figsize=(12, 12))

    i = idx
    img = images[i]
    
    # Convert labels to text
    true_label = class_names[labels[i]] if class_names else labels[i]
    pred_label = class_names[preds[i]] if class_names else preds[i]
    
    
    plt.imshow(img)
    plt.axis("off")
    plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=40, weight='bold')
    
    plt.tight_layout()
    plt.show()


def extract_from_dataset(ds):
    images_list = []
    labels_list = []

    for batch in ds:
        if isinstance(batch, tuple):
            imgs, lbls = batch
        else:
            imgs = batch
            lbls = None

        images_list.append(imgs.numpy())
        if lbls is not None:
            labels_list.append(lbls.numpy())

    images = np.concatenate(images_list, axis=0)

    if labels_list:
        labels = np.concatenate(labels_list, axis=0)
        return images, labels
    else:
        return images


from tensorflow.keras.models import Model
def get_feature_maps(model, layer_name, img):
    """
    model: your LSACT model
    layer_name: e.g. "stem_act"
    img: single RGB image, e.g. shape (600, 600, 3)
    """
    # Sub-model from original input to chosen layer
    sub_model = Model(
        inputs=model.input,
        outputs=model.get_layer(layer_name).output
    )

    # Add batch dimension if needed
    if img.ndim == 3:
        img_batch = np.expand_dims(img, axis=0)
    else:
        img_batch = img

    # Do NOT divide by 255 here; model has Rescaling(1/255.0)
    fmaps = sub_model.predict(img_batch)
    return fmaps   # shape (1, 64, 64, 128) for stem


def plot_feature_maps(fmaps, max_channels=32, title_prefix="Ch"):
    """
    fmaps: (1, H, W, C)
    """
    fm = fmaps[0]                # (H, W, C)
    H, W, C = fm.shape
    num_channels = min(C, max_channels)

    cols = 8
    rows = int(np.ceil(num_channels / cols))

    plt.figure(figsize=(2 * cols, 2 * rows))
    for i in range(num_channels):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(fm[..., i], cmap="viridis")
        plt.axis("off")
        #plt.title(f"{title_prefix} {i}", fontsize=7)
    plt.tight_layout()
    plt.show()


def load_image_from_class(root_dir, class_name, index=0):
    """
    root_dir: path to 'train/' folder
    class_name: a folder inside train/, e.g. 'class3'
    index: which image to load from that folder
    """
    class_dir = os.path.join(root_dir, class_name)
    img_files = sorted(os.listdir(class_dir))

    # pick the image by index
    img_path = os.path.join(class_dir, img_files[index])

    # load
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)   # uint8 shape (H, W, 3)
    return img, img_path



