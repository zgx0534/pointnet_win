# -*- coding: UTF-8 -*-
import argparse
import math
import h5py
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys

#添加环境变量
#os.path.dirname()功能为去掉文件名，返回目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
#sys.path.append(BASE_DIR)帮助程序在本项目文件夹下找依赖
sys.path.append(BASE_DIR)
#os.path.join是路径拼接函数
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))

import provider
import tf_util

#用于命令参数获取，使程序更加友好，个性化运行
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--max_epoch', type=int, default=250, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

#存储从命令行获取的参数，如未指定，则使用默认参数
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

#动态导入网络模板，这样可以灵活的根据需求导入模块，不需要写在文件首部，默认导入pointnet_cls
MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')
#建立log文件夹
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
#把模板文件和训练文件拷进log文件夹
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train.py %s' % (LOG_DIR)) # bkp of train procedure
#打开log日志文件写入基本信息，LOG_FOUT为操作log文件句柄
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')
MAX_NUM_POINT = 2048
NUM_CLASSES = 40

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

#获取主机名
HOSTNAME = socket.gethostname()

# provider.getDataFiles将储存训练文件名的文件转化成字符串数组['file1','file2','file3','file4',……]
TRAIN_FILES = provider.getDataFiles( \
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/train_files.txt'))
TEST_FILES = provider.getDataFiles(\
    os.path.join(BASE_DIR, 'data/modelnet40_ply_hdf5_2048/test_files.txt'))

#记录并同时输出log信息的功能
def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)



# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
# decayed_learning_rate = 0.001         * 0.7        ^ (batch*32    / 20000)
# learning_rate开始是 0.001 随着batch不断增加而减小，batch*32==20000时，learning_rate为0.001*0.7
def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.  0.001
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.  20000
                        DECAY_RATE,          # Decay rate.  0.7
                        staircase=True)

    learning_rate = tf.maximum(learning_rate, 0.00001) # 设置learning_rate最低值
    return learning_rate        

#动量函数
def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,   #0.5
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,   #20000
                      BN_DECAY_DECAY_RATE,   #0.5
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)   # bn_decay不能大于0.99
    return bn_decay

# 训练方法，每次调用它
def train():
    # tf.Graph()表示实例化了一个类，一个用于tensorflow计算和表示用的数据流图
    # 通俗来讲就是：在代码中添加的操作（画中的结点）和数据（画中的线条）都是画在纸上的“画”，而图就是呈现这些画的纸，你可以利用很多线程生成很多张图，但是默认图就只有一张。
    # tf.Graph().as_default()表示将这个类实例，也就是新生成的图作为整个tensorflow运行环境的默认图，如果只有一个主线程不写也没有关系.
    with tf.Graph().as_default():
        # 在0号CPU上执行下边的操作
        with tf.device('/gpu:'+str(GPU_INDEX)):
            # placeholder为占位符，只规定数据的格式要求
            # pointclouds_pl, labels_pl也是placeholder类型
            # BATCH_SIZE=32, NUM_POINT=1024
            # pointclouds_pl：Tensor("Placeholder:0", shape=(32, 1024, 3), dtype=float32, device=/device:GPU:0)
            # labels_pl：Tensor("Placeholder_1:0", shape=(32,), dtype=int32, device=/device:GPU:0)
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # 请注意将参数 global_step = batch 最小化。
            # 这告诉优化器每次训练时都会为你帮助增加'batch'参数。
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            #用来显示标量信息,用于tensorboard图谱
            tf.summary.scalar('bn_decay', bn_decay)

            # 使用自定义的类方法来定义神经网络，这一步直接输出预测结果
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=bn_decay)
            # pred:(32, 40)
            # end_points为64*64的单位阵拉成的向量
            # pred: Tensor("fc3/BiasAdd:0", shape=(32, 40), dtype=float32, device=/device:GPU:0)
            # end_point: {'transform': <tf.Tensor 'transform_net2/Reshape_1:0' shape=(32, 64, 64) dtype=float32>}
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE)
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            # 根据用户设置来选择优化器
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # 创建一个Saver对象，用于保存所有参数和变量 用于迭代训练和测试，需要执行save方法
            saver = tf.train.Saver()
            
        # Create a session
        # tf.ConfigProto()用于在创建session之前，对session配置
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        # merged = tf.merge_all_summaries()
        # merge_all 可以将所有summary全部保存到磁盘，以便tensorboard显示。
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                  sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))

        # Init variables
        init = tf.global_variables_initializer()
        # To fix the bug introduced in TF 0.12.1 as in
        # http://`stackoverflow.com/questions/41543774/invalidargumenterror-for-tensor-bool-tensorflow-0-12-1
        #sess.run(init)
        sess.run(init, {is_training_pl: True})

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch}
        # 训练250组,每10组输出一次
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_one_epoch(sess, ops, train_writer)
            eval_one_epoch(sess, ops, test_writer)
            
            # Save the variables to disk.
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"))
                log_string("Model saved in file: %s" % save_path)



def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # train_file_idxs存储TRAIN_FILES文件的索引 从0到个数的一个排列
    train_file_idxs = np.arange(0, len(TRAIN_FILES))
    # 打乱train_file_idxs的顺序
    np.random.shuffle(train_file_idxs)

    # 一个文件
    for fn in range(len(TRAIN_FILES)):
        log_string('----' + str(fn) + '-----')
        # TRAIN_FILES[0-4]每一个都是保存数据集，前四个文件都是2048个点云，每个点云的尺寸是(2048*3),但是只使用了1024个点
        # 数据集是字典类型, 有四个key, 分别是[u'data', u'faceId', u'label', u'normal']
        # current_data, current_label是一个数据字典的全部数据和标签
        current_data, current_label = provider.loadDataFile(TRAIN_FILES[train_file_idxs[fn]])
        current_data = current_data[:,0:NUM_POINT,:]
        # 打乱current_data, current_label的顺序被打乱 _为索引
        current_data, current_label, _ = provider.shuffle_data(current_data, np.squeeze(current_label))
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        total_correct = 0
        total_seen = 0
        loss_sum = 0
       
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE
            # 通过旋转和抖动来增加批量点云
            # 先取得32个点云形成一个小训练集，然后进行旋转和抖动。
            rotated_data = provider.rotate_point_cloud(current_data[start_idx:end_idx, :, :])
            jittered_data = provider.jitter_point_cloud(rotated_data)
            feed_dict = {ops['pointclouds_pl']: jittered_data,
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training,}
            summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
            train_writer.add_summary(summary, step)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += loss_val
            # =============   测试代码  =================

            with tf.variable_scope('transform_net1/transform_feat') as sc:
                weights = tf.get_variable('weights', [256, 9],
                                  dtype=tf.float32)
                print weights
            # =============   测试代码  =================
        
        log_string('mean loss: %f' % (loss_sum / float(num_batches)))
        log_string('accuracy: %f' % (total_correct / float(total_seen)))

        
def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    total_correct_class = [0 for _ in range(NUM_CLASSES)]
    
    for fn in range(len(TEST_FILES)):
        log_string('----' + str(fn) + '-----')
        current_data, current_label = provider.loadDataFile(TEST_FILES[fn])
        current_data = current_data[:,0:NUM_POINT,:]
        current_label = np.squeeze(current_label)
        
        file_size = current_data.shape[0]
        num_batches = file_size // BATCH_SIZE
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * BATCH_SIZE
            end_idx = (batch_idx+1) * BATCH_SIZE

            feed_dict = {ops['pointclouds_pl']: current_data[start_idx:end_idx, :, :],
                         ops['labels_pl']: current_label[start_idx:end_idx],
                         ops['is_training_pl']: is_training}
            summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
                ops['loss'], ops['pred']], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            correct = np.sum(pred_val == current_label[start_idx:end_idx])
            total_correct += correct
            total_seen += BATCH_SIZE
            loss_sum += (loss_val*BATCH_SIZE)
            for i in range(start_idx, end_idx):
                l = current_label[i]
                total_seen_class[l] += 1
                total_correct_class[l] += (pred_val[i-start_idx] == l)
            
    log_string('eval mean loss: %f' % (loss_sum / float(total_seen)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class,dtype=np.float))))
         


if __name__ == "__main__":
    train()
    LOG_FOUT.close()
