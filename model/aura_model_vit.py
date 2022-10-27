import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from keras_vision_transformer import swin_layers
from keras_vision_transformer import transformer_layers

class PositionEncoding(layers.Layer):
    def __init__(self, model_dim, **kwargs):
        self._model_dim = model_dim
        super(PositionEncoding, self).__init__(**kwargs)

    def call(self, inputs):
        seq_length = inputs.shape[1]
        position_encodings = np.zeros((seq_length, self._model_dim))
        for pos in range(seq_length):
            for i in range(self._model_dim):
                position_encodings[pos, i] = pos / np.power(10000, (i-i % 2) / self._model_dim)
        position_encodings[:, 0::2] = np.sin(position_encodings[:, 0::2]) # 2i
        position_encodings[:, 1::2] = np.cos(position_encodings[:, 1::2]) # 2i+1
        position_encodings = K.cast(position_encodings, 'float32')
        return position_encodings

    def compute_output_shape(self, input_shape):
        return input_shape


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Sampling_Cat(layers.Layer):
    def __init__(self, number_of_classes=5, straight_through=False):
        super(Sampling_Cat, self).__init__()
        self.number_of_classes = number_of_classes
        self.tau = tf.Variable(1.0, name="temperature", trainable=False)
        self.hard = straight_through

    def call(self, inputs):
        z = tfp.distributions.RelaxedOneHotCategorical(self.tau, inputs).sample()
        if self.hard:
            z_hard = tf.cast(tf.one_hot(tf.argmax(z, -1), self.number_of_classes), z.dtype)
            z = tf.stop_gradient(z_hard - z) + z
        return z



class TransformerEncoder(layers.Layer):
    def __init__(self, key_dim=256, num_heads=4, ff_dim=4, rate=0.1):
        super(TransformerEncoder, self).__init__()
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate

        #  Attention and Normalization
        self.att = layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, dropout=rate)
        self.dropout = layers.Dropout(rate)
        self.layernorm = layers.LayerNormalization(epsilon=1e-6)

        # Feed Forward Part
        self.conv_ff_0 = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")
        self.conv_ff_1 = layers.Conv1D(filters=ff_dim, kernel_size=1)

    def call(self, inputs):
        #  Attention and Normalization
        x = self.att(inputs, inputs)
        x = self.dropout(x)
        x = self.layernorm(x)
        res = x + inputs

        # Feed Forward Part
        x = self.conv_ff_0(res)
        x = self.dropout(x)
        x = self.conv_ff_1(x)
        x = self.layernorm(x)
        return x + res


    def get_config(self):
        return {"key_dim": self.key_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "rate": self.rate,
                }


class Transformer(layers.Layer):
    def __init__(self, key_dim, num_heads, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.key_dim = key_dim
        self.num_heads = num_heads

        self.trans_block1 = TransformerEncoder(self.key_dim, self.num_heads)
        self.trans_block2 = TransformerEncoder(self.key_dim, self.num_heads)
        self.trans_block3 = TransformerEncoder(self.key_dim, self.num_heads)
        self.trans_block4 = TransformerEncoder(self.key_dim, self.num_heads)

        self.dense = layers.Dense(self.key_dim, activation='relu')
        self.pos_encoding = PositionEncoding(self.key_dim)
        self.dropout = layers.Dropout(0.1)


    def call(self, inputs):
        x = self.dense(inputs)
        x_pos_enc = self.pos_encoding(x)
        x = x + x_pos_enc
        x = self.dropout(x)
        x = self.trans_block1(x)
        x = self.trans_block2(x)
        x = self.trans_block3(x)
        x = self.trans_block4(x)
        return x

    def get_config(self):
        config = super(Transformer, self).get_config()
        return config


def get_swin(input_size=(100, 100, 1), num_heads=8, embed_dim=64, num_mlp=256, qkv_bias=True, qk_scale=None):
    # Shift-window parameters
    window_size = 2  # Size of attention window (height = width)
    shift_size = window_size // 2  # Size of shifting (shift_size < window_size)
    patch_size = (input_size[0]//10, input_size[1]//10)
    num_patch_x = input_size[0] // patch_size[0]
    num_patch_y = input_size[1] // patch_size[1]

    swin = tf.keras.Sequential([
                    tf.keras.Input(input_size),
                    tf.keras.layers.Rescaling(1.0 / 255),
                    transformer_layers.patch_extract(patch_size),
                    transformer_layers.patch_embedding(num_patch_x * num_patch_y, embed_dim),
                    swin_layers.SwinTransformerBlock(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads,
                                                             window_size=window_size, shift_size=0, num_mlp=num_mlp,
                                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                             mlp_drop=0.01, attn_drop=0.01, proj_drop=0.01,
                                                             drop_path_prob=0.01,
                                                             name='swin_block{}'.format(0)),
                    swin_layers.SwinTransformerBlock(dim=embed_dim, num_patch=(num_patch_x, num_patch_y), num_heads=num_heads,
                                                             window_size=window_size, shift_size=shift_size, num_mlp=num_mlp,
                                                             qkv_bias=qkv_bias, qk_scale=qk_scale,
                                                             mlp_drop=0.01, attn_drop=0.01, proj_drop=0.01,
                                                             drop_path_prob=0.01,
                                                             name='swin_block{}'.format(1)),
                    transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(0)),
                    tf.keras.layers.GlobalAveragePooling1D(),
                    ])
    return swin






class CustomViT(layers.Layer):
    def __init__(self, timesteps=10, **kwargs):
        super(CustomViT, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.vit = get_swin()
        self.timedist = layers.TimeDistributed(self.vit)
        self.lstm = tf.keras.layers.LSTM(self.timesteps, return_sequences=False, stateful=False, dropout=0.1)
        #self.pool = layers.GlobalAveragePooling1D()

    def call(self, inputs):
        features = self.timedist(inputs)
        features = self.lstm(features)
        #features = self.pool(features)
        return features

    def get_config(self):
        config = super(CustomViT, self).get_config()
        return config


class Encoder(layers.Layer):
    def __init__(self, timesteps=10, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.timesteps = timesteps
        self.model_map = CustomViT(timesteps=self.timesteps)

        self.conv_tra = layers.Conv1D(32, kernel_size=3, strides=1, padding='same')
        self.dense_tra = layers.Dense(64, activation='relu', name='x_dense')
        self.trans_tra = Transformer(key_dim=4, num_heads=8)
        self.pool_tra = layers.GlobalAveragePooling1D()
        self.dense_concat = layers.Dense(16, activation='relu')

    def call(self, inputs):
        # map
        x_map = self.model_map(inputs[0])

        # tra
        x_tra = self.conv_tra(inputs[1])
        x_tra = self.dense_tra(x_tra)
        x_tra = self.trans_tra(x_tra)
        x_tra = self.pool_tra(x_tra)

        x_concat = layers.concatenate([x_tra, x_map])
        x_concat = self.dense_concat(x_concat)

        return x_concat

    def get_config(self):
        config = super(Encoder, self).get_config()
        return config


def nnelu(input):
    #Calculate non-negative ELU
    return tf.add(tf.constant(1, dtype=tf.float32), tf.nn.elu(input))


'''
'Model type: '
'0=aura_model_gaus_non_para, '
'1=aura_model_gaus_para, '
'2=aura_model_gum_non_para, '
'3=aura_model_gum_para'
'4=aura_model_gmm_non_para, '
'5=aura_model_gmm_para')
'''

def kl_divergence_two_gauss(mean1,sig1,mean2,sig2):
    return tf.reduce_mean(tf.reduce_sum(tf.math.log(sig2) - tf.math.log(sig1) + ((tf.math.square(sig1) + tf.math.square(mean1-mean2)) / (2*tf.math.square(sig2))) - 0.5, axis=1))




def get_model(obs_seq=10, pred_seq=150, number_of_outputs=2, latent_dim=2, row=32, column=32, training=False, categories=10, type=0):
    # Add activation function to keras
    tf.keras.utils.get_custom_objects().update({'nnelu': tf.keras.layers.Activation(nnelu)})

    if training:
        ##################### Trainings Model #####################

        input_tra_x = tf.keras.layers.Input(shape=(obs_seq, 2), name='encoder_x')
        input_map_x = tf.keras.layers.Input(shape=(obs_seq, row, column, 1), name='input_map_x')

        input_tra_y = tf.keras.layers.Input(shape=(pred_seq, 2), name='input_tra_y')
        input_map_y = tf.keras.layers.Input(shape=(pred_seq, row, column, 1), name='input_map_y')

        # Encoder_x
        encoder_x = Encoder(name='encoder_x_train', timesteps=obs_seq)([input_map_x, input_tra_x])

        # Encoder_y
        encoder_y = Encoder(name='encoder_y_train', timesteps=pred_seq)([input_map_y, input_tra_y])

        # concat
        x = layers.concatenate([encoder_x, encoder_y], name='concat_xy')

        # gaussian latent space
        if type == 0 or type == 1:
            x_vae = layers.Dense(512, activation="relu", name='dense_vae_0')(x)
            x_vae = layers.Dense(256, activation="relu", name='dense_vae_1')(x_vae)
            z_mean = layers.Dense(latent_dim, activation='linear', name='dense_z_mean')(x_vae)
            z_log_var = layers.Dense(latent_dim, activation='linear', name='dense_z_log_var')(x_vae)
            z = Sampling()([z_mean, z_log_var])

        # gumbel latent space
        elif type == 2 or type == 3:
            x_vae = layers.Dense(512, activation="relu", name='dense_vae_0')(x)
            x_vae = layers.Dense(256, activation="relu", name='dense_vae_1')(x_vae)
            logits = layers.Dense(categories, name='dense_vae_2')(x_vae)

            # kl_type == 'categorical' or straight_through:
            z = Sampling_Cat(number_of_classes=categories, straight_through=True)(logits)

        # gmm latent space
        elif type == 4 or type == 5:
            # y_block
            y_hidden = tf.keras.Sequential([
                layers.Dense(1024, activation='elu'),
                layers.Dropout(rate=0.2),
                layers.Dense(128, activation='elu')])(x)

            y_logits = layers.Dense(latent_dim, activation=None, name='y_dense')(y_hidden)
            noise = tf.random.uniform(shape=[latent_dim])
            tau = tf.Variable(1.0, name="temperature", trainable=False)
            y = tf.nn.softmax((y_logits - tf.math.log(-tf.math.log(noise))) / tau, axis=1)  # gumbel softmax

            # Z prior block
            z_prior_mean = layers.Dense(latent_dim, activation=None, name='z_prior_mean')(y)
            # kernel_initializer=tf.initializers.TruncatedNormal(),bias_initializer=tf.keras.initializers.constant(1))
            z_prior_sig = layers.Dense(latent_dim, activation='softplus', name='z_prior_sig',
                                       bias_initializer=tf.keras.initializers.constant(1))(y)

            # Encoder block
            h_top = layers.Dense(512, activation='elu', name='htop')(y)
            h = layers.Dropout(rate=0.2)(x)
            h = layers.Dense(512, activation='elu', name='h')(h)
            h = h + h_top
            z_mean = layers.Dense(latent_dim, name='z_mean', activation=None)(h)
            # kernel_initializer=tf.initializers.TruncatedNormal(),bias_initializer=tf.keras.initializers.constant(1))
            z_sig = layers.Dense(latent_dim, name='z_sig', activation='softplus', bias_initializer=tf.keras.initializers.constant(1))(h)
            z = Sampling()((z_mean, z_sig))


        # concatenate the z and x_encoded_dense
        z = layers.Dense(16, activation="softmax", name='dense_z')(z)
        z_x = layers.concatenate([z, encoder_x], name='concat_enc_dec')

        # DECODER
        x_ = layers.Dense(256, activation='relu', name='decoder_dense_0')(z_x)
        x = layers.RepeatVector(pred_seq, name='decoder_repeat_0')(x_)

        x = layers.LSTM(pred_seq, return_sequences=True, stateful=False, dropout=0.15, activation='tanh',
                        name='decoder_lstm_0')(x)

        #x = layers.GRU(pred_seq, return_sequences=True, stateful=False, dropout=0.15, activation='tanh',
         #               name='decoder_lstm_0')(x)

        if type == 0 or type == 2 or type == 4:
            x = layers.TimeDistributed(layers.Dense(number_of_outputs, name='decoder_dense_1'),
                                       name='decoder_time_dist_0')(x)  # (12, 2)

            model = tf.keras.Model(inputs=[input_map_x, input_tra_x, input_map_y, input_tra_y], outputs=x)

        elif type == 1 or type == 3 or type == 5:
            mean = layers.TimeDistributed(
                layers.Dense(number_of_outputs, name='decoder_dense_1', activation="linear"),
                name='decoder_time_dist_0')(x)  # (12, 2)

            std = layers.TimeDistributed(
                layers.Dense(number_of_outputs, name='decoder_dense_2', activation="nnelu"),
                name='decoder_time_dist_1')(x)  # (12, 2)
            corr = layers.TimeDistributed(
                layers.Dense(number_of_outputs - 1, name='decoder_dense_3', activation="tanh"),
                name='decoder_time_dist_2')(x)  # (12, 2)
            output = layers.Concatenate()([mean, std, corr])
            model = tf.keras.Model(inputs=[input_map_x, input_tra_x, input_map_y, input_tra_y], outputs=output)


        ############################################### loss ###################################################
        # gaussian latent space
        if type == 0 or type == 1:
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            model.add_loss(kl_loss)

        # gumbel latent space
        elif type == 2 or type == 3:
            logits_py = tf.ones_like(logits) * (1. / categories)
            q_cat_z = tfp.distributions.OneHotCategorical(logits=logits)
            p_cat_z = tfp.distributions.OneHotCategorical(logits=logits_py)
            KL_qp = tfp.distributions.kl_divergence(q_cat_z, p_cat_z)
            model.add_loss(tf.reduce_sum(KL_qp))

        # gmm latent space
        elif type == 4 or type == 5:
            # add loss -> kl loss
            x_kl_loss = kl_divergence_two_gauss(z_mean, z_sig, z_prior_mean, z_prior_sig)
            model.add_loss(x_kl_loss)

            py = tf.nn.softmax(y_logits, axis=1)
            y_kl_loss = tf.reduce_mean(tf.reduce_sum(py * (tf.math.log(py + 1e-8) - tf.math.log(1.0 / latent_dim)), axis=1))
            model.add_loss(y_kl_loss)

        # add metrics
        model.mse_tracker = tf.keras.metrics.Mean(name="mse")
        model.dkl_tracker = tf.keras.metrics.Mean(name="dkl")

        # nll tracker for parametric models
        if type == 1 or type == 3 or type == 5:
            model.nll_tracker = tf.keras.metrics.Mean(name="nll")
        return model

    else:
        ##################### Inference Model #####################

        input_tra_x = tf.keras.layers.Input(shape=(obs_seq, 2), name='input_tra_x')
        input_map_x = tf.keras.layers.Input(shape=(obs_seq, row, column, 1), name='input_map_x')
        if type == 0 or type == 1:
            input_decoder = tf.keras.layers.Input(shape=(latent_dim), name='input_dec')
        else:
            input_decoder = tf.keras.layers.Input(shape=(categories), name='input_dec')


        # Encoder_x
        encoder_x = Encoder(name='encoder_x', timesteps=obs_seq)([input_map_x, input_tra_x])

        if type == 4 or type == 5:
            z_prior_mean = layers.Dense(latent_dim, activation=None, name='z_prior_mean')(input_decoder)
            z_prior_sig = layers.Dense(latent_dim, activation='softplus', name='z_prior_sig',
                                       bias_initializer=tf.keras.initializers.constant(1))(input_decoder)
            z_x = tf.random.normal(shape=(tf.shape(z_prior_mean)[0], tf.shape(z_prior_mean)[1]), mean=z_prior_mean,
                                 stddev=z_prior_sig, dtype=tf.float32)
            z = layers.Dense(16, activation="softmax", name='dense_z')(z_x)
        else:
            z = layers.Dense(16, activation="softmax", name='dense_z')(input_decoder)

        # concatenate the z and x_encoded_dense
        z_x = layers.concatenate([z, encoder_x], name='concat_enc_dec')

        # DECODER
        x = layers.Dense(256, activation='relu', name='decoder_dense_0')(z_x)
        x = layers.RepeatVector(pred_seq, name='decoder_repeat_0')(x)
        x = layers.LSTM(pred_seq, return_sequences=True, stateful=False, dropout=0.15, activation='tanh',
                        name='decoder_lstm_0')(x)

        #x = layers.GRU(pred_seq, return_sequences=True, stateful=False, dropout=0.15, activation='tanh',
         #               name='decoder_lstm_0')(x)

        if type == 0 or type == 2 or type == 4:
            x = layers.TimeDistributed(layers.Dense(number_of_outputs, name='decoder_dense_1'),
                                       name='decoder_time_dist_0')(x)  # (12, 2)

            model = tf.keras.Model(inputs=[input_map_x, input_tra_x, input_decoder], outputs=x)

        elif type == 1 or type == 3 or type == 5:
            mean = layers.TimeDistributed(
                layers.Dense(number_of_outputs, name='decoder_dense_1', activation="linear"),
                name='decoder_time_dist_0')(x)  # (12, 2)

            std = layers.TimeDistributed(
                layers.Dense(number_of_outputs, name='decoder_dense_2', activation="nnelu"),
                name='decoder_time_dist_1')(x)  # (12, 2)
            corr = layers.TimeDistributed(
                layers.Dense(number_of_outputs - 1, name='decoder_dense_3', activation="tanh"),
                name='decoder_time_dist_2')(x)  # (12, 2)


            output = layers.Concatenate()([mean, std, corr])
            model = tf.keras.Model(inputs=[input_map_x, input_tra_x, input_decoder], outputs=output)

        return model




def test(model=None, type=0, batch_size=1, pred_length=12, agents=10, map=None, groundtruth=None,
         last_pos=None, latent_dim=2, components=10, output_dim=2):

    #print("Start predicting")

    ################################################ Model 0 Gaussian non-para #########################################
    if type == 0:
        predictions = []
        for batch in range(batch_size):
            for agent in range(agents):
                z_sample = np.random.rand(1, latent_dim)
                pred = model.predict([map[batch:batch + 1], groundtruth[batch:batch + 1], z_sample])
                pred = pred + last_pos[batch]
                predictions.append(pred)
        return np.reshape(predictions, [-1, agents, pred_length, 2])

    ################################################ Model 1 Gaussian para #############################################
    if type == 1:
        std = np.zeros((batch_size, agents, pred_length, 2))
        predictions = []
        for batch in range(batch_size):
            for agent in range(agents):
                z_sample = np.random.rand(1, latent_dim)
                pred = model.predict([map[batch:batch + 1], groundtruth[batch:batch + 1], z_sample])
                mean = pred[:, :, :output_dim]
                std[batch, agent] = pred[:, :, output_dim:2 * output_dim]
                pred = mean + last_pos[batch]
                predictions.append(pred)
        return np.reshape(predictions, [-1, agents, pred_length, 2]), std

    ################################################ Model 2 Gumbel non-para ###########################################
    if type == 2:
        predictions = []
        z = np.eye(components)
        for batch in range(batch_size):
            for agent in range(agents):
                z_sample = np.reshape(z[agent], (1, components))
                pred = model.predict([map[batch:batch + 1], groundtruth[batch:batch + 1], z_sample])
                pred = pred + last_pos[batch]
                predictions.append(pred)
        return np.reshape(predictions, [-1, agents, pred_length, 2])


    ################################################ Model 3 Gumbel para ###############################################
    if type == 3:
        std = np.zeros((batch_size, agents, pred_length, 2))
        predictions = []
        for batch in range(batch_size):
            z = np.eye(components)
            for agent in range(agents):
                z_sample = np.reshape(z[agent], (1, components))
                pred = model.predict([map[batch:batch + 1], groundtruth[batch:batch + 1], z_sample])
                mean = pred[:, :, :output_dim]  # (batch, timesteps, number_of_outputs)
                std[batch, agent] = pred[:, :, output_dim:2*output_dim]
                pred = mean + last_pos[batch]
                predictions.append(pred)
        return np.reshape(predictions, [-1, agents, pred_length, 2]), std

    ################################################ Model 4 GMM Non-para #######################################################
    if type == 4:
        predictions = []
        z = np.eye(components)
        for batch in range(batch_size):
            for agent in range(agents):
                z_sample = np.reshape(z[agent], (1, components))
                pred = model.predict([map[batch:batch + 1], groundtruth[batch:batch + 1], z_sample])
                pred = pred + last_pos[batch]
                predictions.append(pred)
        return np.reshape(predictions, [-1, agents, pred_length, 2])

    ################################################ Model 5 GMM para ##################################################
    if type == 5:
        std = np.zeros((batch_size, agents, pred_length, 2))
        predictions = []
        for batch in range(batch_size):
            z = np.eye(components)
            for agent in range(agents):
                z_sample = np.reshape(z[agent], (1, components))
                pred = model.predict([map[batch:batch + 1], groundtruth[batch:batch + 1], z_sample])
                mean = pred[:, :, :output_dim]  # (batch, timesteps, number_of_outputs)
                std[batch, agent] = pred[:, :, output_dim:2 * output_dim]
                pred = mean + last_pos[batch]
                predictions.append(pred)
        return np.reshape(predictions, [-1, agents, pred_length, 2]), std