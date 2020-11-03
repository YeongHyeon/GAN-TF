import os
import numpy as np
import tensorflow as tf
import source.layers as lay

class GAN(object):

    def __init__(self, \
        height, width, channel, ksize, zdim, \
        learning_rate=1e-3, path='', verbose=True):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel, self.ksize, self.zdim = \
            height, width, channel, ksize, zdim
        self.learning_rate = learning_rate
        self.path_ckpt = path

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel], \
            name="x")
        self.z = tf.compat.v1.placeholder(tf.float32, [None, self.zdim], \
            name="z")
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[], \
            name="batch_size")
        self.training = tf.compat.v1.placeholder(tf.bool, shape=[], \
            name="training")

        self.layer = lay.Layers()

        self.variables, self.losses = {}, {}
        self.__build_model(x_real=self.x, z=self.z, ksize=self.ksize, verbose=verbose)
        self.__build_loss()

        with tf.control_dependencies(self.variables['ops_d']):
            self.optimizer_d = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate, name='Adam_d').minimize(\
                self.losses['loss_d'], var_list=self.variables['params_d'])

        with tf.control_dependencies(self.variables['ops_g']):
            self.optimizer_g = tf.compat.v1.train.AdamOptimizer( \
                self.learning_rate*5, name='Adam_g').minimize(\
                self.losses['loss_g'], var_list=self.variables['params_g'])

        tf.compat.v1.summary.scalar('GAN/loss_d', self.losses['loss_d'])
        tf.compat.v1.summary.scalar('GAN/loss_g', self.losses['loss_g'])
        self.summaries = tf.compat.v1.summary.merge_all()

        self.__init_session(path=self.path_ckpt)

    def step(self, x, z, iteration=0, training=False):

        feed_tr = {self.x:x, self.z:z, self.batch_size:x.shape[0], self.training:True}
        feed_te = {self.x:x, self.z:z, self.batch_size:x.shape[0], self.training:False}

        summary_list = []
        if(training):
            try:
                _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)

                _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                    feed_dict=feed_tr, options=self.run_options, run_metadata=self.run_metadata)
                summary_list.append(summaries)
            except:
                _, summaries = self.sess.run([self.optimizer_d, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)

                _, summaries = self.sess.run([self.optimizer_g, self.summaries], \
                    feed_dict=feed_tr)
                summary_list.append(summaries)

            for summaries in summary_list:
                self.summary_writer.add_summary(summaries, iteration)

        x_fake, loss_d, loss_g = \
            self.sess.run([self.variables['g_fake'], self.losses['loss_d'], self.losses['loss_g']], \
            feed_dict=feed_te)

        outputs = {'x_fake':x_fake, 'loss_d':loss_d, 'loss_g':loss_g}
        return outputs

    def save_parameter(self, model='model_checker', epoch=-1):

        self.saver.save(self.sess, os.path.join(self.path_ckpt, model))
        if(epoch >= 0): self.summary_writer.add_run_metadata(self.run_metadata, 'epoch-%d' % epoch)

    def load_parameter(self, model='model_checker'):

        path_load = os.path.join(self.path_ckpt, '%s.index' %(model))
        if(os.path.exists(path_load)):
            print("\nRestoring parameters")
            self.saver.restore(self.sess, path_load.replace('.index', ''))

    def confirm_params(self, verbose=True):

        print("\n* Parameter arrange")

        ftxt = open("list_parameters.txt", "w")
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if(verbose): print(text)
            ftxt.write("%s\n" %(text))
        ftxt.close()

    def confirm_bn(self, verbose=True):

        print("\n* Confirm Batch Normalization")

        t_vars = tf.compat.v1.trainable_variables()
        for var in t_vars:
            if('bn' in var.name):
                tmp_x = np.zeros((1, self.height, self.width, self.channel))
                tmp_z = np.zeros((1, self.zdim))
                values = self.sess.run(var, \
                    feed_dict={self.x:tmp_x, self.z:tmp_z, self.batch_size:1, self.training:False})
                if(verbose): print(var.name, var.shape)
                if(verbose): print(values)

    def __init_session(self, path):

        try:
            sess_config = tf.compat.v1.ConfigProto()
            sess_config.gpu_options.allow_growth = True
            self.sess = tf.compat.v1.Session(config=sess_config)

            self.sess.run(tf.compat.v1.global_variables_initializer())
            self.saver = tf.compat.v1.train.Saver()

            self.summary_writer = tf.compat.v1.summary.FileWriter(path, self.sess.graph)
            self.run_options = tf.compat.v1.RunOptions(trace_level=tf.compat.v1.RunOptions.FULL_TRACE)
            self.run_metadata = tf.compat.v1.RunMetadata()
        except: pass

    def __build_loss(self):

        tmp_real = tf.reduce_mean( \
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits( \
            logits=self.variables['d_real'], labels=tf.ones_like(self.variables['d_real'])))
        tmp_fake = tf.reduce_mean( \
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits( \
            logits=self.variables['d_fake'], labels=tf.zeros_like(self.variables['d_fake'])))

        self.losses['loss_d'] = tmp_real + tmp_fake
        self.losses['loss_g'] = tf.reduce_mean( \
            tf.compat.v1.nn.sigmoid_cross_entropy_with_logits( \
            logits=self.variables['d_fake'], labels=tf.ones_like(self.variables['d_fake'])))

        self.variables['params_d'], self.variables['params_g'] = [], []
        for var in tf.compat.v1.trainable_variables():
            text = "Trainable: " + str(var.name) + str(var.shape)
            if('dis_' in var.name): self.variables['params_d'].append(var)
            elif('gen_' in var.name): self.variables['params_g'].append(var)

        self.variables['ops_d'], self.variables['ops_g'] = [], []
        for ops in tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS):
            if('dis_' in ops.name): self.variables['ops_d'].append(ops)
            elif('gen_' in ops.name): self.variables['ops_g'].append(ops)

    def __build_model(self, x_real, z, ksize=3, verbose=True):

        if(verbose): print("\n* Discriminator")
        self.variables['d_real'] = \
            self.__encoder(x=x_real, ksize=ksize, reuse=False, \
            name='dis', verbose=verbose)

        if(verbose): print("\n* Generator")
        self.variables['g_fake'] = \
            self.__decoder(z=z, ksize=ksize, reuse=False, \
            name='gen', verbose=verbose)

        self.variables['d_fake'] = \
            self.__encoder(x=self.variables['g_fake'], ksize=ksize, reuse=True, \
            name='dis', verbose=False)

    def __encoder(self, x, ksize=3, reuse=False, name='enc', activation='relu', verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            conv1_1 = self.layer.conv2d(x=x, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 1, 16], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_1" %(name), verbose=verbose)
            conv1_2 = self.layer.conv2d(x=conv1_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 16], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_2" %(name), verbose=verbose)
            maxp1 = self.layer.maxpool(x=conv1_2, ksize=2, strides=2, padding='SAME', \
                name="%s_pool1" %(name), verbose=verbose)

            conv2_1 = self.layer.conv2d(x=maxp1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 32], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_1" %(name), verbose=verbose)
            conv2_2 = self.layer.conv2d(x=conv2_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 32], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_2" %(name), verbose=verbose)
            maxp2 = self.layer.maxpool(x=conv2_2, ksize=2, strides=2, padding='SAME', \
                name="%s_pool2" %(name), verbose=verbose)

            conv3_1 = self.layer.conv2d(x=maxp2, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_1" %(name), verbose=verbose)
            conv3_2 = self.layer.conv2d(x=conv3_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_2" %(name), verbose=verbose)

            rs = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, int(7*7*64)], \
                name="%s_rs" %(name))
            e = self.layer.fully_connected(x=rs, c_out=1, \
                batch_norm=False, training=self.training, \
                activation=None, name="%s_fc1" %(name), verbose=verbose)

            return e

    def __decoder(self, z, ksize=3, reuse=False, name='dec', activation='relu', verbose=True):

        with tf.variable_scope(name, reuse=reuse):

            fc1 = self.layer.fully_connected(x=z, c_out=7*7*64, \
                batch_norm=True, training=self.training, \
                activation=activation, name="%s_fc1" %(name), verbose=verbose)
            rs = tf.compat.v1.reshape(fc1, shape=[self.batch_size, 7, 7, 64], \
                name="%s_rs" %(name))

            convt1_1 = self.layer.conv2d(x=rs, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_1" %(name), verbose=verbose)
            convt1_2 = self.layer.conv2d(x=convt1_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 64, 64], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv1_2" %(name), verbose=verbose)

            convt2_1 = self.layer.convt2d(x=convt1_2, stride=2, padding='SAME', \
                output_shape=[self.batch_size, 14, 14, 32], filter_size=[ksize, ksize, 32, 64], \
                dilations=[1, 1, 1, 1], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_1" %(name), verbose=verbose)
            convt2_2 = self.layer.conv2d(x=convt2_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 32, 32], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv2_2" %(name), verbose=verbose)

            convt3_1 = self.layer.convt2d(x=convt2_2, stride=2, padding='SAME', \
                output_shape=[self.batch_size, 28, 28, 16], filter_size=[ksize, ksize, 16, 32], \
                dilations=[1, 1, 1, 1], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_1" %(name), verbose=verbose)
            convt3_2 = self.layer.conv2d(x=convt3_1, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, 16], batch_norm=True, training=self.training, \
                activation=activation, name="%s_conv3_2" %(name), verbose=verbose)
            d = self.layer.conv2d(x=convt3_2, stride=1, padding='SAME', \
                filter_size=[ksize, ksize, 16, self.channel], batch_norm=False, training=self.training, \
                activation='sigmoid', name="%s_conv3_3" %(name), verbose=verbose)

            return d
