import tensorflow as tf
from tensorflow.keras import layers, models, activations

# Custom BiasPositionalEmbedding layer
def layer_norm(inputs, name=None):
    """ Layer Normalization """
    return layers.LayerNormalization(axis=-1, epsilon=1e-5, name=name)(inputs)


class BiasPositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, axis=[1, 2], attn_height=-1, initializer="zeros", **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.initializer = initializer
        self.attn_height = attn_height

    def build(self, input_shape):
        bb_shape = (input_shape[1], input_shape[2], 1)
        self.bb = self.add_weight("positional_embedding", shape=bb_shape, initializer=self.initializer, trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        return inputs + self.bb

    def get_config(self):
        config = super().get_config()
        config.update({"axis": self.axis, "attn_height": self.attn_height})
        return config

def ct_stem(inputs, stem_width, activation="gelu", name="", num_layers=2, **kwargs):

    def depthwise_pointwise_conv_bn(x, filters, activation="gelu", name=""):
        """Applies depthwise and pointwise convolutions followed by batch normalization and activation.

        Args:
            x: Input tensor to the block.
            filters: Number of filters for the pointwise convolution.
            activation: Activation function to use.
            name: Name prefix for the layers.

        Returns:
            Output tensor after the convolutions and activation.
        """
        dw = layers.DepthwiseConv2D(kernel_size=5, strides=1, padding="same", use_bias=False, name=name + "dw")(x)
        dw = batchnorm_with_activation(dw, activation=activation, act_first=True, name=name + "dw_")

        pw = layers.Conv2D(filters, kernel_size=1, strides=1, padding="same", use_bias=False, name=name + "pw")(dw)
        pw = batchnorm_with_activation(pw, activation=activation, act_first=True, name=name + "pw_")

        return pw

    nn = layers.Conv2D(stem_width, kernel_size=1, strides=1, padding="same", use_bias=False, name=name + "1_pw")(inputs)
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "1_pw_")

    for i in range(num_layers):
        nn = depthwise_pointwise_conv_bn(nn, stem_width, activation=activation, name=name + f"{i + 2}_")

    return nn

# Attention block
# Custom convolutional layer without bias
def conv2d_no_bias(inputs, filters, kernel_size=1, strides=1, padding="same", use_bias=False, name=None):
    """Custom Conv2D layer without bias."""
    return tf.keras.layers.Conv2D(
        filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name
    )(inputs)


# Custom depthwise convolutional layer
def depthwise_conv2d_no_bias(kernel_size, strides=1, padding="same", use_bias=False, name=None):
    """ Custom depthwise convolutional layer without bias """
    return layers.DepthwiseConv2D(kernel_size, strides=strides, padding=padding, use_bias=use_bias, name=name)


# Custom batch normalization layer with optional activation
def batchnorm_with_activation(inputs, activation="relu", act_first=False, name=None):
    """Batch normalization layer with optional activation"""
    # Apply batch normalization
    nn = layers.BatchNormalization(name=name + "_bn")(inputs)

    # Apply activation if specified
    if activation:
        nn = activation_by_name(nn, activation=activation, name=name + "_act")
    return nn


# Activation by name
def activation_by_name(inputs, activation="gelu", name=""):
    """ Custom activation by name """
    if activation.lower() == "gelu":
        return layers.Activation(tf.nn.gelu, name=name)(inputs)
    else:
        return layers.Activation(activation, name=name)(inputs)

# Define the custom DecoupledFullyConnectedAttentionBlock
class DecoupledFullyConnectedAttentionBlock(layers.Layer):
    def __init__(self, out_channel, name=""):
        super(DecoupledFullyConnectedAttentionBlock, self).__init__(name=name)
        self.out_channel = out_channel

        # Define layers during initialization
        self.avg_pool = layers.AvgPool2D(pool_size=1, strides=1)
        self.conv1 = layers.Conv2D(self.out_channel, kernel_size=1, use_bias=True, name=self.name + "conv1_")
        self.bn1 = layers.BatchNormalization(name=self.name + "bn1_")
        self.act1 = layers.Activation("relu", name=self.name + "act1_")  # Assuming ReLU or other activation
        self.dw_conv2 = layers.DepthwiseConv2D(kernel_size=(5, 1), use_bias=True, name=self.name + "dwconv2_")
        self.bn2 = layers.BatchNormalization(name=self.name + "bn2_")
        self.act2 = layers.Activation("relu", name=self.name + "act2_")
        self.dw_conv3 = layers.DepthwiseConv2D(kernel_size=(1, 5), use_bias=True, name=self.name + "dwconv3_")
        self.bn3 = layers.BatchNormalization(name=self.name + "bn3_")
        self.act3 = layers.Activation("relu", name=self.name + "act3_")
        self.sigmoid = layers.Activation("sigmoid", name=self.name + "sigmoid_")

    def call(self, inputs):
        # Average Pooling
        nn = self.avg_pool(inputs)

        # Convolution without bias
        nn = self.conv1(nn)
        nn = self.bn1(nn)
        nn = self.act1(nn)

        # Depthwise Convolutions
        nn = self.dw_conv2(nn)
        nn = self.bn2(nn)
        nn = self.act2(nn)
        nn = self.dw_conv3(nn)
        nn = self.bn3(nn)
        nn = self.act3(nn)

        # Activation
        nn = self.sigmoid(nn)

        # Resizing
        size = tf.shape(inputs)[1:-1] if tf.keras.backend.image_data_format() == "channels_last" else tf.shape(inputs)[2:]
        nn = tf.image.resize(nn, size, method='nearest')  # Resize with nearest neighbor interpolation

        return nn

# Deep Inverted residual feed-forward block
def deep_inverted_residual_feed_forward(inputs, num_layers=3, expansion=4, activation="gelu",
                                        name="", use_bn=True, kernel_size=3, dropout_rate=0.0):
    """Deep Inverted Residual Feed Forward Block (IRFFN).

    Args:
        inputs: Input tensor to the feed forward block.
        num_layers: Number of layers in the feed forward block.
        expansion: Expansion factor for the intermediate channels.
        activation: Activation function to use (default: "gelu").
        name: Name prefix for the layers.
        use_bn: Whether to use batch normalization after convolutions.
        kernel_size: Size of the depth-wise convolution kernel.
        dropout_rate: Dropout rate to apply after each layer.

    Returns:
        Output tensor after passing through the feed forward block.
    """

    # Determine channel axis based on image data format
    channel_axis = -1 if tf.keras.backend.image_data_format() == "channels_last" else 1
    in_channel = inputs.shape[channel_axis]
    output = inputs

    for i in range(num_layers):
        # Expand the number of channels with a point-wise convolution
        expanded = tf.keras.layers.Conv2D(int(in_channel * expansion), kernel_size=1, use_bias=True,
                                           name=f"{name}pointwise_expand_{i}_")(output)
        if use_bn:
            expanded = tf.keras.layers.BatchNormalization(name=f"{name}pointwise_expand_{i}_bn")(expanded)
        expanded = tf.keras.layers.Activation(activation, name=f"{name}pointwise_expand_{i}_act")(expanded)
        if dropout_rate > 0.0:
            expanded = tf.keras.layers.Dropout(dropout_rate)(expanded)

        # Depth-wise convolution and residual connection
        dw = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, padding="same", use_bias=True,
                                              name=f"{name}depthwise_{i}_")(expanded)
        dw = tf.keras.layers.Add(name=f"{name}depthwise_{i}_add")([expanded, dw])
        if use_bn:
            dw = tf.keras.layers.BatchNormalization(name=f"{name}depthwise_{i}_bn")(dw)
        dw = tf.keras.layers.Activation(activation, name=f"{name}depthwise_{i}_act")(dw)
        if dropout_rate > 0.0:
            dw = tf.keras.layers.Dropout(dropout_rate)(dw)

        # Project back to original number of channels
        output = tf.keras.layers.Conv2D(in_channel, kernel_size=1, use_bias=True,
                                         name=f"{name}pointwise_project_{i}_")(dw)
        if use_bn:
            output = tf.keras.layers.BatchNormalization(name=f"{name}pointwise_project_{i}_bn")(output)

    return output

def LPU(inputs, name, activation='relu', dropout_rate=0.2):
    """Local Pattern Unit (LPU) using point-wise and depth-wise convolutions"""

    # Point-wise convolution (1x1)
    pw_conv = layers.Conv2D(inputs.shape[-1], kernel_size=1, padding="same", use_bias=False, name=name + "lpu_pw")(inputs)
    pw_conv = layers.BatchNormalization(name=name + "lpu_pw_bn")(pw_conv)
    pw_conv = layers.Activation(activation, name=name + "lpu_pw_activation")(pw_conv)

    # Depth-wise convolution (3x3)
    dw_conv = layers.DepthwiseConv2D(kernel_size=3, padding="same", use_bias=False, name=name + "lpu_dw")(pw_conv)
    dw_conv = layers.BatchNormalization(name=name + "lpu_dw_bn")(dw_conv)
    dw_conv = layers.Activation(activation, name=name + "lpu_dw_activation")(dw_conv)

    # Optional dropout for regularization
    if dropout_rate > 0:
        dw_conv = layers.Dropout(dropout_rate, name=name + "lpu_dw_dropout")(dw_conv)

    # Squeeze-and-Excitation Block
    se = layers.GlobalAveragePooling2D(name=name + "se_pool")(inputs)
    se = layers.Dense(inputs.shape[-1] // 16, activation='relu', name=name + "se_fc1")(se)  # Squeeze
    se = layers.Dense(inputs.shape[-1], activation='sigmoid', name=name + "se_fc2")(se)  # Excitation
    se = layers.Reshape((1, 1, inputs.shape[-1]), name=name + "se_reshape")(se)
    dw_conv = layers.Multiply(name=name + "se_scale")([dw_conv, se])  # Scale

    # Add the input to the output of the LPU (skip connection)
    lpu_out = layers.Add(name=name + "lpu_out")([inputs, dw_conv])

    return lpu_out

# LSACT Block
def ct_block(inputs, num_heads=4, sr_ratio=1, expansion=4, qkv_bias=False, attn_use_bn=False,
              attn_out_bias=False, activation="gelu", drop_rate=0, name=""):

    # Using LPU method
    lpu_out = LPU(inputs, name=name + "lpu_", activation=activation, dropout_rate=0.2)

    # Attention Block
    attn = DecoupledFullyConnectedAttentionBlock(num_heads, name=name + "attn_")(lpu_out)

    attn_out = layers.Multiply(name=name + "attn_out")([lpu_out, attn])

    # Feed Forward Network (FFN)
    ffn = deep_inverted_residual_feed_forward(attn_out, num_layers=1, expansion=expansion,
                                              activation=activation, name=name + "ffn_", use_bn=attn_use_bn)

    ffn_out = layers.Add(name=name + "ffn_output")([attn_out, ffn])

    return ffn_out

def LSACT(
    num_blocks,
    out_channels,
    stem_width=128,
    num_heads=[1, 1, 4, 8],
    sr_ratios=[1, 2, 4, 8],
    ffn_expansion=4,
    qkv_bias=False,
    attn_out_bias=False,
    attn_use_bn=False,
    feature_activation=None,
    feature_act_first=True,
    input_shape=(64, 64, 3),
    num_classes=100,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    output_num_features=1280,
    dropout=0,
    model_name="LSACT",
    kwargs=None,
    target_size = 64,
):
    inputs = layers.Input(shape=(None, None, 3))
    x = layers.Resizing(target_size, target_size)(inputs)
    x = layers.Rescaling(1.0/255.0, name="rescale")(x)
    
    # Improved STEM Block, replacement of ct_stem, more efficient
    x = layers.Conv2D(3, 4, strides=4, padding="same", name="stem")(inputs)
    x = activations.gelu(x); 
    nn = layers.BatchNormalization(name="stem_bn")(x)
    

    #nn = ct_stem(inputs, stem_width=stem_width, activation=activation, name="stem_")
    #nn = x
    total_blocks = sum(num_blocks)
    global_block_id = 0

    for stack_id, (num_block, out_channel, num_head, sr_ratio) in enumerate(zip(num_blocks, out_channels, num_heads, sr_ratios)):
        stage_name = "stack{}_".format(stack_id + 1)
        nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=1, use_bias=True, name=stage_name + "down_sample")
        nn = layer_norm(nn, name=stage_name)

        block_pos_emb = BiasPositionalEmbedding(axis=[1, 2], attn_height=nn.shape[1], name=stage_name + "pos_emb")
        
        #nn = block_pos_emb(nn)
        
        for block_id in range(num_block):
            name = stage_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = ct_block(nn, num_head, sr_ratio, ffn_expansion, qkv_bias, attn_use_bn, attn_out_bias, activation, drop_rate=block_drop_rate, name=name)


    if output_num_features > 0:
        nn = conv2d_no_bias(nn, output_num_features, 1, strides=2, use_bias=True, name="features_")
        feature_activation = activation if feature_activation is None else feature_activation
        nn = batchnorm_with_activation(nn, activation=feature_activation, act_first=feature_act_first, name="features_")

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def LSACT_Tiny(input_shape=(160, 160, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    """
    Define the LSACT-Tiny model in TensorFlow.

    Args:
    - input_shape: Shape of the input images. Default is (160, 160, 3).
    - num_classes: Number of output classes for the classification head. Default is 1000.
    - activation: Activation function for the intermediate layers. Default is "gelu".
    - classifier_activation: Activation function for the classifier head. Default is "softmax".
    - **kwargs: Additional arguments to pass to the LSACT model.

    Returns:
    - model: A TensorFlow model instance.
    """
    num_blocks = [2, 4]           # Number of blocks in each stage
    out_channels = [64, 128]       # Output channels for each stage
    stem_width = 128                # Width of the stem block
    ffn_expansion = 4          # Expansion ratio for the feed-forward network

    # Call the LSACT model function with specific configurations for LSACT-Tiny
    return LSACT(
        num_blocks=num_blocks,
        out_channels=out_channels,
        stem_width=stem_width,
        ffn_expansion=ffn_expansion,
        input_shape=input_shape,
        num_classes=num_classes,
        activation=activation,
        classifier_activation=classifier_activation,
        model_name="LSACT_Tiny",
        **kwargs
    )

