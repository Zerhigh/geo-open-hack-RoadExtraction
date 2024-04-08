import tensorflow as tf
import numpy as np
import os
import cv2
from tqdm import tqdm
from keras.callbacks import Callback
import json
from keras_unet_collection import models, base, utils, losses
from PIL import Image
import time
import datetime
import random
from utils import create_folder

# enabling mixed precision, only works on GPU capability 7.0
# currently used GPU has version 6.1
# from keras.mixed_precision import experimental as mixed_precision
# tf.keras.mixed_precision.set_global_policy("mixed_float16")
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)


def hybrid_loss(y_true, y_pred):
    """
        receives: true input tensor, prdiction input tensor
        returns: hybrid loss
        calculates the hybrid loss from FTL and IOU, adapted from keras_unet_collection
    """
    # focal tversky for class imbalance, iou for crossentropy binary
    loss_focal = losses.focal_tversky(y_true, y_pred, alpha=0.7, gamma=4 / 3)
    loss_iou = losses.iou_seg(y_true, y_pred)

    return loss_focal + loss_iou  # +loss_ssim


class CheckpointsCallback(Callback):
    # Callback Class, adapted from https://github.com/divamgupta/image-segmentation-keras

    def __init__(self, checkpoints_path, model_name, pred_images, pred_shapei, pred_loaded_model, pred_name,
                 patience=5):
        self.checkpoints_path = checkpoints_path
        self.patience = patience
        # best_weights to store the weights at which the minimum loss occurs.
        self.best_weights = None
        self.metrics = {}
        self.model_name = model_name
        self.start_lr = 1e-4
        self.stopping_epoch = 0

        # for predicting results at each epochs end
        self.pred_images = pred_images
        self.pred_shapei = pred_shapei
        self.pred_loaded_model = pred_loaded_model
        self.pred_name = pred_name

    def on_train_begin(self, logs=None):
        """
            starts on beginning of each epoch, allocates variables
        """
        # define log dict
        for metric in logs:
            print(f"here are the metrics {metric}")
            self.metrics[metric] = {}
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best = np.Inf  # 0#np.Inf  # 0

    def on_epoch_end(self, epoch, logs=None):
        """
            updates variables and Callbacks at the end of the epoch
        """
        # Old Early Stopping 6a)
        # if epoch % 10 == 0 and epoch >= 30:
        #     self.start_lr = self.start_lr / 10
        #     print('adjusted lr to ', self.start_lr)
        #     tf.keras.backend.set_value(self.model.optimizer.learning_rate, self.start_lr)

        # save weights
        if self.checkpoints_path is not None:
            self.model.save_weights(f'{self.checkpoints_path}/checkpoints.{str(epoch)}')
            print("saved ", self.checkpoints_path + "." + str(epoch))

        # access metrics and losses
        try:
            current_val = logs.get(f'val_{self.model_name}_output_final_activation_iou_seg') + logs.get(
                f'val_{self.model_name}_output_final_activation_focal_tversky')
        except:
            print('didnt find metrics')
            current_val = logs.get("val_loss")
        print("logs:", logs)

        # compare losses with previous epoch to compare early stopping
        if np.less(current_val, self.best):  # np.greater(current_val, self.best): #np.less(current_val, self.best):
            print(f'achieved better results.. {current_val}')
            self.best = current_val
            self.wait = 0
            # Record the best weights if current results is better (less).
            self.best_weights = self.model.get_weights()
            self.stopping_epoch = epoch
        else:
            self.wait += 1
            print(f'achieved worse results than {self.best} with {current_val}')
            if self.wait >= self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print(f"Restoring model weights from the end of the best epoch {self.stopping_epoch}.")
                self.model.set_weights(self.best_weights)

        # logging data into metric
        for metric in logs:
            if metric in self.metrics:
                self.metrics[metric].append(logs.get(metric))
            else:
                self.metrics[metric] = [logs.get(metric)]

        # log epoch number to metric
        self.metrics['epoch'] = self.stopped_epoch

        # predict 10 imgs on epoch end to visually compare performance
        for arr in tqdm(self.pred_images)[:10]:
            out = np.empty((1, self.pred_shapei, self.pred_shapei, 3))
            with Image.open(f'{filepath}images/{arr}') as pixio:
                pix = pixio.resize((self.pred_shapei, self.pred_shapei), Image.Resampling.NEAREST)
                out[0, ...] = np.array(pix)[..., :3]

            image = out / 255.
            out = self.pred_loaded_model.predict(x=image, verbose=0)
            img_out = np.array(out[-1]).reshape((self.shapei, self.pred_shapei, 1))
            img_out2 = cv2.normalize(img_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
            cv2.imwrite(f"{out_path}/{self.pred_name}/epoch_results/e{epoch}_{arr}", img_out2)

    def on_train_end(self, logs=None):
        """
            wegen training ends, save logs to a file
        """

        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        # saving log to file
        self.model.set_weights(self.best_weights)
        with open(f'{self.checkpoints_path}/{self.model_name}_logs.txt', 'w') as log_file:
            log_file.write(json.dumps(str(self.metrics)))


def augment_image(image):
    """
        receives: image
        returns: augmented image
        applies random augmentations to images when loading in the dataset
    """
    # update seed
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
    # Random augmentations.
    image = tf.image.stateless_random_brightness(image, max_delta=0.5, seed=new_seed)
    image = tf.image.stateless_random_contrast(image, 0.2, 0.5, new_seed)
    image = tf.image.stateless_random_flip_left_right(image, new_seed)
    image = tf.image.stateless_random_flip_up_down(image, new_seed)

    return image


def create_dataset(base, BATCH_SIZE=2, TRAIN_VAL_TEST_SPLIT=[0.8, 0.2]):
    """
        receives: base path
        returns: training and validation datasets
        creates datasets for training sem. segmentation with tensorflow
    """

    image_dir = base + 'images/'
    label_dir = base + 'rehashed_ones/'
    # define the paths
    image_files = tf.data.Dataset.list_files(os.path.join(image_dir, "*.png"), shuffle=True, seed=69)
    label_files = tf.data.Dataset.list_files(os.path.join(label_dir, "*.png"), shuffle=True, seed=69)

    # combine datasets
    dataset = tf.data.Dataset.zip((image_files, label_files))
    # apply shuffling
    dataset = dataset.shuffle(buffer_size=BATCH_SIZE)

    # create map of dataset to apply preprocessing and image import
    dataset = dataset.map(
        map_func=load_and_preprocess_image_and_label,
        num_parallel_calls=tf.data.AUTOTUNE
    )

    # split dataset
    dataset_size = tf.data.experimental.cardinality(dataset).numpy()
    print(f'loaded dataset {dataset} with size {dataset_size}')
    train_size = int(TRAIN_VAL_TEST_SPLIT[0] * dataset_size)
    val_size = int(TRAIN_VAL_TEST_SPLIT[1] * dataset_size)

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)

    # batch datasets
    train_dataset = train_dataset.batch(BATCH_SIZE)
    val_dataset = val_dataset.batch(BATCH_SIZE)

    # prefetch datasets for better memory handling
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    return train_dataset, val_dataset


def load_and_preprocess_image_and_label(image_path, label_path):
    """
        receives: image path, mask path
        returns: augmented image, mask both as tensor objects
        reads files and applies augmentations
    """
    # load images
    image = tf.io.read_file(image_path)
    label = tf.io.read_file(label_path)
    image = tf.image.decode_png(image, channels=3)
    label = tf.image.decode_png(label, channels=1)  # channels=3

    # normalize image
    image = tf.image.per_image_standardization(image)  # tf.cast(image, tf.float32) / 255.0

    # apply augmentation on image
    return augment_image(image), label  # image_pre, label_pre #image, label


def train_unet2d(filepath, out_path, model_name, train=True, predict=True, save_model=True, load_model=False,
                 load_model_name=None, train_on='GPU', verbose=True, super_verbose=False):

    physical_devices = tf.config.list_physical_devices(f'/device:{train_on}:0')
    print(physical_devices)

    # using models, losses and metrics from https://github.com/yingkaisha/keras-unet-collection/blob/main/examples/user_guide_models.ipynb
    with tf.device(f'/{train_on}:0'):
        start_script = time.time()
        start_time = datetime.datetime.now()

        checkpoints_path = f'{out_path}/{model_name}/checkpoints'

        # create prediction sample for each epoch
        imgs = os.listdir(f'{filepath}images/')
        random.seed(86)
        random.shuffle(imgs)

        # creating folder structure
        folders = [f'{out_path}/{model_name}', f'{out_path}/{model_name}/model', f'{out_path}/{model_name}/results',
                   f'{out_path}/{model_name}/checkpoints', f'{out_path}/{model_name}/weights',
                   f'{out_path}/{model_name}/epoch_results']

        for folder in folders:
            create_folder(folder_name=folder, verbose=super_verbose)

        # define hyperparamters
        #IMAGE_SIZE = [512, 512]
        #RESHAPE_SIZE = [256, 256]
        shape_i = 512  # 256
        n_labels = 1
        epochs = 200
        lr_start = 1e-4

        BATCH_SIZE = 2
        TRAIN_VAL_TEST_SPLIT = [0.8, 0.2]  # if a test split has been done on file before, apply this: [0.8889, 0.1111]
        PATIENCE = 7
        TF_FORCE_GPU_ALLOW_GROWTH = True

        if train_on == 'GPU':
            # GPU enhancment
            TF_GPU_ALLOCATOR = 'cuda_malloc_async'
            try:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
            except:
                # Invalid device or cannot modify virtual devices once initialized.
                pass

        # create the dataset
        if train:
            train_dataset, val_dataset = create_dataset(filepath, BATCH_SIZE=BATCH_SIZE,
                                                        TRAIN_VAL_TEST_SPLIT=TRAIN_VAL_TEST_SPLIT)

        # model params
        type_m = 'unet_2d'
        backbone = 'DenseNet201'  # 'EfficientNetB1' #'DenseNet121' # 'ResNet50V2' "DenseNet201",
        weights = 'imagenet'

        filter_num_down = [32, 64, 128, 256, 512]
        filter_num_skip = [32, 32, 32, 32]
        filter_num_aggregate = 160
        stack_num_down = 2
        stack_num_up = 2
        filter_num = [64, 128, 256, 512, 1024]
        activation = 'ReLU'
        output_activation = 'Sigmoid'
        batch_norm = True
        pool = True
        unpool = True
        freeze_backbone = True
        freeze_batch_norm = True

        # compiling
        loss = hybrid_loss
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_start)
        metrics = [hybrid_loss, losses.iou_seg]
        model_parameters = {'type_m': type_m, 'shape_i': shape_i, 'activation': activation,
                            'filter_num_down': filter_num_down, 'filter_num_skip': filter_num_skip,
                            'filter_num_aggregate': filter_num_aggregate, 'output_activation': output_activation,
                            'stack_num_down': stack_num_down, 'stack_num_up': stack_num_up,
                            'filter_num': filter_num, 'batch_norm': batch_norm, 'pool': pool, 'unpool': unpool,
                            'backbone': backbone, 'weights': weights, 'freeze_backbone': freeze_backbone,
                            'freeze_batch_norm': freeze_batch_norm, 'loss': 'hybrid_loss',
                            'optimizer': 'tf.keras.optimizers.Adam(learning_rate:lr_start)', 'metrics': 'hybrid_loss'}

        with open(f'{out_path}/{model_name}/model_parameters.txt', 'w') as log_file:
            log_file.write(json.dumps(model_parameters))

        # unet basic
        loaded_model = models.unet_2d((shape_i, shape_i, 3), filter_num=filter_num,
                                      n_labels=n_labels,
                                      stack_num_down=stack_num_down, stack_num_up=stack_num_up,
                                      activation=activation,
                                      output_activation=output_activation,
                                      batch_norm=batch_norm, pool=pool, unpool=unpool,
                                      weights=weights, backbone=backbone,  # 'ResNet50V2'
                                      freeze_backbone=freeze_backbone, freeze_batch_norm=freeze_batch_norm,
                                      name=model_name)

        # load existing model
        if load_model and load_model_name is not None:
            print("loading model")
            loading_model = utils.dummy_loader(f'{out_path}/{load_model_name}/model/')
            loaded_model.set_weights(loading_model)
            lr_middle = 1e-6
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_middle)

        # compile model
        loaded_model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        # add callbacks
        callbacks = [CheckpointsCallback(checkpoints_path=checkpoints_path, model_name=model_name, patience=PATIENCE,
                                         pred_images=imgs, pred_shapei=shape_i, pred_loaded_model=loaded_model,
                                         pred_name=model_name),
                    tf.keras.callbacks.ReduceLROnPlateau(patience=PATIENCE-2, monitor='val_loss', factor=0.4, verbose=1)]

        # output model summary
        loaded_model.summary()

        # train model
        if train:
            unet3plus_history = loaded_model.fit(x=train_dataset, validation_data=val_dataset, batch_size=BATCH_SIZE, epochs=epochs, verbose=1, callbacks=callbacks) # , callbacks=callbacks

        # save model
        if save_model:
            print(f'saving model to {out_path}/{model_name}/model/')
            loaded_model.save(f'{out_path}/{model_name}/model/', save_traces=True)

            # save model weights
            loaded_model.save_weights(f'{out_path}/{model_name}/weights/')
            print(f'saved weights to {out_path}/{model_name}/weights/')

        # predict segmentation output
        if predict:
            print(f'predicting output to {out_path}/{model_name}/results/')
            print(f'{filepath}images/')

            for arr in tqdm(os.listdir(f'{filepath}images/')):
                out = np.empty((1, shape_i, shape_i, 3))
                with Image.open(f'{filepath}images/{arr}') as pixio:
                    pix = pixio.resize((shape_i, shape_i), Image.Resampling.NEAREST)
                    out[0, ...] = np.array(pix)[..., :3]

                image = out / 255.
                out = loaded_model.predict(x=image, verbose=0)

                img_out = np.array(out[-1]).reshape((shape_i, shape_i, 1))
                img_out2 = cv2.normalize(img_out, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                cv2.imwrite(f"{out_path}/{model_name}/results/{arr}", img_out2)

        print(f'script took {round(time.time() - start_script, 2)}s from {start_time} to {datetime.datetime.now()}')

    return


verbose = True
super_verbose = False

# static seed in script
rng = tf.random.Generator.from_seed(123, alg='philox')
seed = rng.make_seeds(2)[0]

fp = 'D:/SHollendonner/'

out_path = f'{fp}/segmentation_results/'
filepath = f'{fp}/tiled512/small_test_sample/'

train_unet2d(out_path=out_path, filepath=filepath, model_name='U-Net_DenseNet201', load_model=True,
             load_model_name='3105_MS_combined', verbose=verbose,
             super_verbose=super_verbose)
