import os
from sklearn.model_selection import KFold
import keras
from keras.layers import Input, Dense, Activation, Bidirectional
from keras.layers import Reshape, Lambda, BatchNormalization, Dropout
from keras import applications
from keras.layers.recurrent import LSTM
from keras.layers.merge import add, concatenate
from keras.models import Model
from keras.optimizers import Adadelta, Adam
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
import numpy as np
from keras import backend as K

import json
import cv2
import os, random
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import itertools
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import argparse
from data import random_padding

parser = argparse.ArgumentParser(description='Ocr training arguments')
parser.add_argument('--datapath', default='/data/cuonghn/data/cinamon/ocr/train/')
parser.add_argument('--label_path', default='/data/cuonghn/data/cinamon/ocr/labels.json')
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--idx', default=0, type=int)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--finetune', default=0, type=int)
parser.add_argument('--pretrained', default=0, type=int)
parser.add_argument('--lr', default=0.001, type=float)
parser.add_argument('--weight_path', default='model_best_0.h5')
parser.add_argument('--old_path', default='old/model_best_0.h5')
parser.add_argument('--wbs_lib', default='/data/cuonghn/project/cinamon/CTCWordBeamSearch/cpp/proj/TFWordBeamSearch.so')
args = parser.parse_args()

"""Random eraser
"""
def get_random_eraser(p=0.5, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, erase_type='random'):
    """
    erase_type: random, white, black
    """
    def eraser(input_img):
        img_h, img_w, _ = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break
        c = 255
        if erase_type == 'random':
            c = np.random.uniform(v_l, v_h)
        elif erase_type == 'black':
            c = 0
        input_img[top:top + h, left:left + w, :] = c
        return input_img
    return eraser


"""Encoding - Decoding
"""
letters = " #'()+,-./:0123456789ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
MAX_LEN = 70
SIZE = 2560, 160
CHAR_DICT = len(letters) + 1

import re
chars = letters
wordChars = "ABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ"
corpus = ' \n '.join(json.load(open(args.label_path)).values())
pattern = re.compile('[^\sABCDEFGHIJKLMNOPQRSTUVWXYabcdeghiklmnopqrstuvxyzÂÊÔàáâãèéêìíòóôõùúýăĐđĩũƠơưạảấầẩậắằẵặẻẽếềểễệỉịọỏốồổỗộớờởỡợụủỨứừửữựỳỵỷỹ_]+')
corpus = pattern.sub('', corpus)
word_beam_search_module = tf.load_op_library(args.wbs_lib)
mat=tf.placeholder(tf.float32)
beamsearch_decoder = word_beam_search_module.word_beam_search(mat, 25, 'Words', 0, corpus, chars, wordChars)


def text_to_labels(text):
    return list(map(lambda x: letters.index(x), text))

def labels_to_text(labels):
    return ''.join(list(map(lambda x: letters[x] if x < len(letters) else "_", labels)))

def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    # the 2 is critical here since the first couple outputs of the RNN
    # tend to be garbage:
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def decode_batch(out):
    ret0 = []
    ret = []
    for j in range(out.shape[0]):
        out_best = list(np.argmax(out[j, 2:], 1))
        ret0.append(labels_to_text(out_best))
        out_best = [k for k, g in itertools.groupby(out_best)]
        outstr = labels_to_text(out_best)
        ret.append(outstr)
    return ret, ret0

def beamsearch(sess, y_pred):
    y_pred = y_pred.transpose((1, 0, 2))
    results = sess.run(beamsearch_decoder, {mat:y_pred})
    blank=len(chars)
    results_text = []
    for res in results:
        s=''
        for label in res:
            if label==blank:
                break
            s+=chars[label] # map label to char
        results_text.append(s)
    return results_text

# Visualize prediction
class VizCallback(keras.callbacks.Callback):
    def __init__(self, y_func, text_img_gen, text_size, sess, num_display_words=6):
        self.y_func = y_func
        self.text_img_gen = text_img_gen
        self.num_display_words = num_display_words
        self.text_size = text_size
        self.sess = sess

    def on_epoch_end(self, epoch, logs={}):
        batch = next(self.text_img_gen.next_batch())[0]
        inputs = batch['the_inputs'][:self.num_display_words]
        labels = batch['the_labels'][:self.num_display_words].astype(np.int32)
        labels = [labels_to_text(label) for label in labels]
        
        pred = self.y_func([inputs])[0]
        pred_beamsearch_texts = beamsearch(self.sess, pred)
        pred_texts, pred_texts0 = decode_batch(pred)
        for i in range(min(self.num_display_words, len(inputs))):
            print(f"Label: {labels[i]}\nUnformat pred: {pred_texts0[i]}\nPredict: {pred_texts[i]}")
            print(f"Word beam search pred: {pred_beamsearch_texts[i]}")

"""
Image Generator
"""
class TextImageGenerator:
    def __init__(self, img_dirpath, labels_path, img_w, img_h,
                 batch_size, downsample_factor, idxs, training=True, max_text_len=9, n_eraser=5):
        self.img_h = img_h
        self.img_w = img_w
        self.batch_size = batch_size
        self.max_text_len = max_text_len
        self.idxs = idxs
        self.downsample_factor = downsample_factor
        self.img_dirpath = img_dirpath                  # image dir path
        self.labels= json.load(open(labels_path))
        self.img_dir = os.listdir(self.img_dirpath)     # images list
        self.img_dir = [self.img_dir[idx] for idx in self.idxs]
        self.n = len(self.img_dir)                      # number of images
        self.indexes = list(range(self.n))
        self.cur_index = 0
        self.imgs = np.zeros((self.n, self.img_h, self.img_w, 3), dtype=np.float32)
        self.training = training
        self.n_eraser = n_eraser
        self.random_eraser = get_random_eraser(s_l=0.004, s_h=0.005, r_1=0.01, r_2=1/0.01, v_l=-128, v_h=128)
        self.texts = []
        image_datagen_args = {
		'shear_range': 0.1,
		'zoom_range': 0.01,
		'width_shift_range': 0.001,
		'height_shift_range': 0.1,
		'rotation_range': 1,
		'horizontal_flip': False,
		'vertical_flip': False
	    }
        self.image_datagen = ImageDataGenerator(**image_datagen_args)

    def build_data(self):
        print(self.n, " Image Loading start...")
        for i, img_file in enumerate(self.img_dir):
            img = cv2.imread(os.path.join(self.img_dirpath, img_file), cv2.IMREAD_GRAYSCALE)
            # Add random padding
            img = random_padding(img, max_width_height_ratio=20, min_width_height_ratio=10, chanels=1)
            # Resize & black white
            img = cv2.resize(img, (self.img_w, self.img_h))
            (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
            
            img = img.astype(np.float32)
            img = preprocess_input(img)
            self.imgs[i] = img
            self.texts.append(self.labels[img_file])
        print("Image Loading finish...")

    def next_sample(self):
        self.cur_index += 1
        if self.cur_index >= self.n:
            self.cur_index = 0
            random.shuffle(self.indexes)
        return self.imgs[self.indexes[self.cur_index]], self.texts[self.indexes[self.cur_index]]

    def next_batch(self):
        while True:
            X_data = np.zeros([self.batch_size, self.img_w, self.img_h, 3], dtype=np.float32)     # (bs, 128, 64, 1)
            Y_data = np.zeros([self.batch_size, self.max_text_len], dtype=np.float32)             # (bs, 9)
            input_length = np.ones((self.batch_size, 1), dtype=np.float32) * (self.img_w // self.downsample_factor - 2)  # (bs, 1)
            label_length = np.zeros((self.batch_size, 1), dtype=np.float32)           # (bs, 1)

            for i in range(self.batch_size):
                img, text = self.next_sample()

                if self.training:
                    params = self.image_datagen.get_random_transform(img.shape)
                    img = self.image_datagen.apply_transform(img, params)
                    for _ in range(self.n_eraser):
                        img = self.random_eraser(img)

                img = img.transpose((1, 0, 2))
                # random eraser if training
                X_data[i] = img
                Y_data[i,:len(text)] = text_to_labels(text)
                label_length[i] = len(text)

            inputs = {
                'the_inputs': X_data,  # (bs, 128, 64, 1)
                'the_labels': Y_data,  # (bs, 8)
                'input_length': input_length,  # (bs, 1)
                'label_length': label_length  # (bs, 1)
            }
            outputs = {'ctc': np.zeros([self.batch_size])}   # (bs, 1)
            yield (inputs, outputs)


"""
Training path
"""
def get_model(input_shape, training, finetune):
    inputs = Input(name='the_inputs', shape=input_shape, dtype='float32')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    inner = base_model(inputs)
    inner = Reshape(target_shape=(int(inner.shape[1]), -1), name='reshape')(inner)
    inner = Dense(512, activation='relu', kernel_initializer='he_normal', name='dense1')(inner) 
    inner = Dropout(0.5)(inner)
    
    # RNN layer
    lstm1_merged = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25), merge_mode='sum')(inner) # (None, 80, 256)
    lstm1_merged = BatchNormalization()(lstm1_merged)
    lstm2_merged = Bidirectional(LSTM(256, return_sequences=True, kernel_initializer='he_normal', name='lstm1', dropout=0.25, recurrent_dropout=0.25), merge_mode='concat')(lstm1_merged)
    lstm = BatchNormalization()(lstm2_merged)
    
    y_pred = Dense(CHAR_DICT, activation='softmax', kernel_initializer='he_normal',name='dense2')(lstm)
    
    labels = Input(name='the_labels', shape=[MAX_LEN], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')

    loss_out = Lambda(ctc_lambda_func, output_shape=(1,), name='ctc')([y_pred, labels, input_length, label_length])

    for layer in base_model.layers:
        layer.trainable = finetune
    
    y_func = K.function([inputs], [y_pred])
    
    if training:
        Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out).summary()
        return Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out), y_func
    else:
        return Model(inputs=[inputs], outputs=y_pred)

def train_kfold(idx, kfold, datapath, epochs, batch_size, finetune, label_path, lr):
    sess = tf.Session()
    K.set_session(sess)
    model, y_func = get_model((*SIZE, 3), training=True, finetune=finetune)
    ada = Adam(lr=lr)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

    ## load data
    train_idx, valid_idx = kfold[idx]
    train_generator = TextImageGenerator(datapath, label_path, *SIZE, batch_size, 32, train_idx, True, MAX_LEN)
    train_generator.build_data()
    valid_generator  = TextImageGenerator(datapath, label_path, *SIZE, batch_size, 32, valid_idx, False, MAX_LEN)
    valid_generator.build_data()

    ## callbacks
#     if not os.path.isdir('model/'):
#         os.makedirs('model')
    weight_path = 'model_best_%d.h5' % idx
    old_weight_path = '../input/crnn-vgg16-pretrained-ctc/model_best_0.h5'
    ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
    early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)
    vis = VizCallback(y_func, valid_generator, len(valid_idx), sess)

    if finetune:
        print('load pretrain model')
        model.load_weights(old_weight_path)

    model.fit_generator(generator=train_generator.next_batch(),
                    steps_per_epoch=int(len(train_idx) / batch_size),
                    epochs=epochs,
                    callbacks=[ckp, vis],
                    validation_data=valid_generator.next_batch(),
                    validation_steps=int(len(valid_idx) / batch_size))
    
def train(epochs, batch_size, lr, finetune=False):
    nsplits = 5
    datapath = '../input/train/sample2/'
    label_path = '../input/labels.json'

    nfiles = np.arange(len(os.listdir(datapath)))

    kfold = list(KFold(nsplits).split(nfiles))
#     for idx in range(nsplits):
    for idx in range(1):
        train_kfold(idx, kfold, datapath, epochs, batch_size, finetune, label_path, lr)


# KFOLD, args
nsplits = 5
os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

datapath = args.datapath
label_path = args.label_path
nfiles = np.arange(len(os.listdir(datapath)))
kfold = list(KFold(nsplits).split(nfiles))
idx = args.idx

epochs = args.epochs
batch_size = args.batch_size
pretrained = args.pretrained
finetune = args.finetune

## load data
train_idx, valid_idx = kfold[idx]
train_generator = TextImageGenerator(datapath, label_path, *SIZE, batch_size, 32, train_idx, True, MAX_LEN)
train_generator.build_data()
valid_generator  = TextImageGenerator(datapath, label_path, *SIZE, batch_size, 32, valid_idx, False, MAX_LEN)
valid_generator.build_data()

# Get model
sess = tf.Session()
K.set_session(sess)
model, y_func = get_model((*SIZE, 3), training=True, finetune=finetune)
ada = Adam(lr=args.lr)
model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=ada)

#weight_path = 'model_best_%d.h5' % idx
weight_path = args.weight_path
ckp = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
vis = VizCallback(y_func, valid_generator, len(valid_idx), sess)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5)

if args.pretrained:
    print(f'Load pretrain model from {args.old_path}')
    model.load_weights(args.old_path)

model.fit_generator(generator=train_generator.next_batch(),
                steps_per_epoch=int(len(train_idx) / batch_size),
                epochs=epochs,
                callbacks=[ckp, vis, early_stop],
                validation_data=valid_generator.next_batch(),
                validation_steps=int(len(valid_idx) / batch_size))

