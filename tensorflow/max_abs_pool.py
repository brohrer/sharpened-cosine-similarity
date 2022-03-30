import tensorflow as tf

class MaxAbsPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size, pad_to_fit=False):
        super(MaxAbsPool2D, self).__init__()
        self.pad = pad_to_fit
        self.pool_size = pool_size

    def compute_output_shape(self, in_shape):
        if self.pad:
            return (in_shape[0],
                    tf.math.ceil(in_shape[1] / self.pool_size),
                    tf.math.ceil(in_shape[2] / self.pool_size),
                    in_shape[3])
        return (in_shape[0],
                (in_shape[1] // self.pool_size),
                (in_shape[2] // self.pool_size),
                in_shape[3])

    def compute_padding(self, in_shape):
        mod_y = in_shape[1] % self.pool_size
        y1 = mod_y // 2
        y2 = mod_y - y1
        mod_x = in_shape[2] % self.pool_size
        x1 = mod_x // 2
        x2 = mod_x - x1
        self.padding = ((0, 0), (y1, y2), (x1, x2), (0, 0))

    def build(self, input_shape):
        self.in_shape = input_shape
        self.out_shape = self.compute_output_shape(self.in_shape)
        self.compute_padding(self.in_shape)

    def stack(self, inputs):
        if self.pad:
            inputs = tf.pad(inputs, self.padding)
        max_height = (tf.shape(inputs)[1] // self.pool_size) * self.pool_size
        max_width = (tf.shape(inputs)[2] // self.pool_size) * self.pool_size
        stack = tf.stack(
            [inputs[:, i:max_height:self.pool_size, j:max_width:self.pool_size, :]
             for i in range(self.pool_size) for j in range(self.pool_size)],
            axis=-1)
        return stack

    @tf.function
    def call(self, inputs, training=None):
        stacked = self.stack(inputs)
        inds = tf.argmax(tf.abs(stacked), axis=-1, output_type=tf.int32)
        ks = tf.shape(stacked)
        idx = tf.stack([
            *tf.meshgrid(
                tf.range(0, ks[0]),
                tf.range(0, ks[1]),
                tf.range(0, ks[2]),
                tf.range(0, ks[3]),
                indexing='ij'
            ), inds],
            axis=-1)
        x = tf.gather_nd(stacked, idx)
        x = tf.reshape(x, (-1, *self.out_shape[1:]))
        return x
