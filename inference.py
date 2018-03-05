import numpy as np
import resnet_model
import tensorflow as tf
import os
import wget

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from PIL import Image
import time
import joblib
cate_dict = joblib.load('./fasion_data/filtered_cate_mapping_dict_08_04')

cate_idx_dict = {}

for key, value in cate_dict.items():
    cate_idx_dict[value] = key

file_name_list = joblib.load('./fasion_data/train_file_name_list_08_04')
class_label_list = joblib.load('./fasion_data/train_class_label_list_08_04')


class Inference:
    def __init__(self):
        batch_size = 1
        num_classes = 33
        
        hps = resnet_model.HParams(batch_size=batch_size,
                                   num_classes=num_classes,
                                   min_lrn_rate=0.0001,
                                   lrn_rate=0.1,
                                   num_residual_units=5,
                                   use_bottleneck=False,
                                   weight_decay_rate=0.0002,
                                   relu_leakiness=0.1,
                                   optimizer='mom')
        images = tf.placeholder(tf.float32, shape=(1, 128, 128, 3))
        labels = tf.placeholder(tf.int32, shape=(1, 33))
        
        self.model = resnet_model.ResNet(hps, images, labels, 'test')
        self.model.build_graph()
        
        
        # self.sess = tf.Session()
        
        self.saver = tf.train.Saver()
        try:
            log_root = 'amazon_log_08_04/'
            self.ckpt_state = tf.train.get_checkpoint_state(log_root)
        except tf.errors.OutOfRangeError as e:
            tf.logging.error('Cannot restore checkpoint: %s', e)
        
        self.string_name = tf.placeholder(tf.string, name='test_string_name')
        # self.file_name_ = tf.placeholder(tf.string, name='test_file_name')
        self.file_name_ = tf.Variable('20180305_112629')
        image_size = 128
        batch_size = 1
        num_classes = 33

        filename_queue = tf.train.string_input_producer([self.file_name_])

        image_reader = tf.WholeFileReader()

        file_path, image_file = image_reader.read(filename_queue)

        image = tf.image.decode_jpeg(image_file, channels=3)

        longer = tf.reduce_max(tf.shape(image))
        image = tf.image.resize_image_with_crop_or_pad(image, longer, longer)
        image = tf.image.resize_images(image, [128, 128], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        image = tf.image.resize_image_with_crop_or_pad(
            image, image_size, image_size)

        temp_image = image

        image = tf.image.per_image_standardization(image)
        label = tf.Variable([1])

        indices = tf.reshape(tf.range(0, batch_size, 1), [batch_size, 1])

        image = tf.expand_dims(image, 0)

        # random label just for test
        label = tf.expand_dims(label, 0)

        label = tf.sparse_to_dense(
            tf.concat(values=[indices, label], axis=1),
            [batch_size, num_classes], 1.0, 0.0)
        
        
        # Attach placeholder
        self.image = image
        self.label = label
        self.temp_image = temp_image
        
    def get_file_name(self, img_url, f_name):
        try:
            file_name = wget.download(img_url, out=f_name)
        except Exception as e:
            print(e)
        return f_name
    
    def inference(self, img_url):
        self.sess = tf.Session()
        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())

        timestr = time.strftime("%Y%m%d_%H%M%S")
        file_name = self.get_file_name(img_url, timestr)
        assign = self.file_name_.assign(file_name)
        self.sess.run(assign)
        self.saver.restore(self.sess, os.getcwd() + '/' + self.ckpt_state.model_checkpoint_path)
        print('model load is done')
        print('\n file_name : ', file_name)
        
        tf.train.start_queue_runners(self.sess)
        
        print('yoyo')
        image, label, temp_image = self.sess.run([self.image, self.label, self.temp_image])
        Image.fromarray(temp_image, 'RGB').save('bbb.jpeg')

        feed_dict = {self.model._images: image, self.model.labels: label}
        predictions, predict_labels, m_images, global_avg_pool = self.sess.run(
            [self.model.predictions, self.model.labels, self.model._images, self.model.global_avg_pool], feed_dict)
        predictions = np.argmax(predictions, axis=1)
        
        return cate_idx_dict[predictions[0]]
