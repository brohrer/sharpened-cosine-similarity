import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers

class CosSim2D(layers.Layer):
    def __init__(
        self,
        n_kernels,
        kernel_size=3,
        padding: int=0,
        stride: int=1,
        depthwise_separable: bool=False,
        p_init: float=.7,
        q_init: float=1.,
        p_scale: float=5.,
        q_scale: float=.3,
        eps: float=1e-6,
    ):
        super(CosSim2D, self).__init__()
        self.depthwise_separable = depthwise_separable
        self.n_kernels = n_kernels
        assert kernel_size in [1, 3, 5], "kernel of this size not supported"
        self.kernel_size = kernel_size
        if self.kernel_size == 1:
            self.stack = lambda x: x
        elif self.kernel_size == 3:
            self.stack = self.stack3x3
        elif self.kernel_size == 5:
            self.stack = self.stack5x5
        self.stride = stride
        self.p_init= p_init
        self.q_init= q_init
        self.p_scale = p_scale
        self.q_scale = q_scale
        self.eps = eps

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_y = math.ceil((self.in_shape[1] - 2*self.clip) / self.stride)
        self.out_x = math.ceil((self.in_shape[2] - 2*self.clip) / self.stride)
        self.flat_size = self.out_x * self.out_y
        self.channels = self.in_shape[3]

        if self.depthwise_separable:
            self.w = self.add_weight(
                shape=(1, tf.square(self.kernel_size), self.n_kernels),
                initializer="glorot_uniform", name='w',
                trainable=True,
            )
        else:
            self.w = self.add_weight(
                shape=(1, self.channels * tf.square(self.kernel_size), self.n_kernels),
                initializer="glorot_uniform", name='w',
                trainable=True,
            )

        # self.b = self.add_weight(
        #     shape=(self.n_kernels,), initializer="zeros", trainable=True, name='b')

        p_initializer = tf.keras.initializers.Constant(
            value=float(self.p_init * self.p_scale))
        q_initializer = tf.keras.initializers.Constant(
            value=float(self.q_init * self.q_scale))
        self.p = self.add_weight(
            shape=(self.n_kernels,),
            initializer=p_initializer,
            trainable=True,
            name='p')
        self.q = self.add_weight(
            shape=(1,),
            initializer=q_initializer,
            trainable=True,
            name='q')

    @tf.function
    def l2_normal(self, x, axis=None, epsilon=1e-12):
        square_sum = tf.reduce_sum(tf.square(x), axis, keepdims=True)
        x_inv_norm = tf.sqrt(tf.maximum(square_sum, epsilon))
        return x_inv_norm

    @tf.function
    def sigplus(self, x):
        return tf.nn.sigmoid(x) * tf.nn.softplus(x)

    @tf.function
    def stack3x3(self, image):
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(image[:, :y-1-self.clip:, :x-1-self.clip, :], tf.constant([[0,0], [self.pad,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],   # top row
                tf.pad(image[:, :y-1-self.clip, self.clip:x-self.clip, :],   tf.constant([[0,0], [self.pad,0], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-1-self.clip, 1+self.clip:, :],  tf.constant([[0,0], [self.pad,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                
                tf.pad(image[:, self.clip:y-self.clip, :x-1-self.clip, :],   tf.constant([[0,0], [0,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],   # middle row
                image[:,self.clip:y-self.clip:self.stride,self.clip:x-self.clip:self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip, 1+self.clip:, :],    tf.constant([[0,0], [0,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                    
                tf.pad(image[:, 1+self.clip:, :x-1-self.clip, :],  tf.constant([[0,0], [0,self.pad], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],    # bottom row
                tf.pad(image[:, 1+self.clip:, self.clip:x-self.clip, :],    tf.constant([[0,0], [0,self.pad], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:, 1+self.clip:, :],   tf.constant([[0,0], [0,self.pad], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:]
            ], axis=3)
        return stack
    
    @tf.function
    def stack5x5(self, image):
        x = tf.shape(image)[2]
        y = tf.shape(image)[1]
        stack = tf.stack(
            [
                tf.pad(image[:, :y-2-self.clip:, :x-2-self.clip, :],          tf.constant([[0,0], [self.pad,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, 1:x-1-self.clip, :],         tf.constant([[0,0], [self.pad,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, self.clip:x-self.clip  , :], tf.constant([[0,0], [self.pad,0], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, 1+self.clip:-1 , :],         tf.constant([[0,0], [self.pad,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, :y-2-self.clip:, 2+self.clip: , :],           tf.constant([[0,0], [self.pad,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
             
                tf.pad(image[:, 1:y-1-self.clip:,  :x-2-self.clip, :],          tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:,  1:x-1-self.clip, :],         tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:,  self.clip:x-self.clip  , :], tf.constant([[0,0], [self.pad_1,self.pad_1], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:, 1+self.clip:-1  , :],         tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1:y-1-self.clip:, 2+self.clip:  , :],           tf.constant([[0,0], [self.pad_1,self.pad_1], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                
                tf.pad(image[:, self.clip:y-self.clip,  :x-2-self.clip, :],      tf.constant([[0,0], [0,0], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip,  1:x-1-self.clip, :],     tf.constant([[0,0], [0,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                       image[:, self.clip:y-self.clip,  self.clip:x-self.clip , :][:,::self.stride,::self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip, 1+self.clip:-1  , :],     tf.constant([[0,0], [0,0], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, self.clip:y-self.clip, 2+self.clip:  , :],       tf.constant([[0,0], [0,0], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                    
                tf.pad(image[:, 1+self.clip:-1,  :x-2-self.clip, :],           tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1,  1:x-1-self.clip, :],          tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1,  self.clip:x-self.clip  , :],  tf.constant([[0,0], [self.pad_1,self.pad_1], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1, 1+self.clip:-1  , :],          tf.constant([[0,0], [self.pad_1,self.pad_1], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 1+self.clip:-1, 2+self.clip:  , :],            tf.constant([[0,0], [self.pad_1,self.pad_1], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
                    
                tf.pad(image[:, 2+self.clip:,  :x-2-self.clip, :],           tf.constant([[0,0], [0,self.pad], [self.pad,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:,  1:x-1-self.clip, :],          tf.constant([[0,0], [0,self.pad], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:,  self.clip:x-self.clip  , :],  tf.constant([[0,0], [0,self.pad], [0,0], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:, 1+self.clip:-1  , :],          tf.constant([[0,0], [0,self.pad], [self.pad_1,self.pad_1], [0,0]]))[:,::self.stride,::self.stride,:],
                tf.pad(image[:, 2+self.clip:, 2+self.clip:  , :],            tf.constant([[0,0], [0,self.pad], [0,self.pad], [0,0]]))[:,::self.stride,::self.stride,:],
            ], axis=3)
        return stack

    def call_body(self, inputs):
        channels = tf.shape(inputs)[-1]
        x = self.stack(inputs)
        x = tf.reshape(
            x, (-1, self.flat_size, channels * tf.square(self.kernel_size)))
        p = tf.exp(self.p / self.p_scale)
        q = tf.exp(-self.q / self.q_scale)

        x_norm = (self.l2_normal(x, axis=2)) + q
        w_norm = (self.l2_normal(self.w, axis=1))
        x = tf.matmul(x / x_norm, self.w / w_norm)

        sign = tf.sign(x)
        x = tf.abs(x) + self.eps
        x = tf.pow(x, self.p)
        x = sign * x
        x = tf.reshape(x, (-1, self.out_y, self.out_x, self.n_kernels))
        return x

    @tf.function
    def call(self, inputs, training=None):
        if self.depthwise_separable:
            channels = tf.shape(inputs)[-1]
            x = tf.vectorized_map(self.call_body, tf.expand_dims(tf.transpose(inputs, (3,0,1,2)), axis=-1))
            s = tf.shape(x)
            x = tf.transpose(x, (1,2,3,4,0))
            x = tf.reshape(x, (-1, self.out_y, self.out_x, self.channels * self.n_kernels))
            return x
        else:
            x = self.call_body(inputs)
            return x
