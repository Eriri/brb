import os
import tensorflow as tf
import numpy as np
from scipy.special import softmax
from scipy.stats import entropy
from sklearn.preprocessing import StandardScaler
from util import kfold, generate_variable
from dataset import dataset_numeric_classification

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

tf.keras.backend.set_floatx('float32')
gpus = tf.config.experimental.list_physical_devices('GPU')
[tf.config.experimental.set_memory_growth(gpu, True) for gpu in gpus]
sw = tf.summary.create_file_writer('logs')
sw.set_as_default()
dtype = tf.float32
eps_mul = tf.constant(1e-3, dtype=dtype)
eps_add = tf.constant(1e-30, dtype=dtype)

class Model(tf.Module):
    def __init__(self, rule_num, att_dim, res_dim, raw_x, raw_y):
        super(Model, self).__init__()
        self.a = tf.Variable(initial_value=raw_x,dtype=dtype)
        self.b = tf.Variable(initial_value=raw_y,dtype=dtype)
        self.d = tf.Variable(initial_value=tf.ones(shape=(att_dim,), dtype=dtype))
        self.r = tf.Variable(initial_value=tf.zeros(shape=(rule_num,), dtype=dtype))

    def gao(self, x):
        w = tf.math.square(self.a - tf.expand_dims(x, -2)) * tf.math.exp(self.d)
        aw = tf.exp(-tf.reduce_sum(w, -1)) * tf.math.exp(self.r)
        bw = aw + eps_mul * tf.expand_dims(tf.reduce_max(aw, -1), -1) + eps_add
        cw = tf.expand_dims(bw / (tf.expand_dims(tf.reduce_sum(bw, -1), -1) - bw), -1)
        bc = tf.reduce_prod(tf.expand_dims(cw, -1) * tf.nn.softmax(self.b) + 1.0, -3) - 1.0 + eps_add
        return bc
    
    def output(self, x):
        bc = self.gao(x)
        return tf.math.log(bc[:, :, 1] / bc[:, :, 0])

    def predict(self, x):
        bc = self.gao(x)
        ez = bc[:, :, 1] / bc[:, :, 0]
        return ez / tf.expand_dims(tf.reduce_sum(ez, -1), -1)

    def evaluate(self, x):
        return tf.math.argmax(self.predict(x), -1)


def creat_model(rule_num, att_dim, res_dim, x=None, y=None):
    if x is None and y is None:
        model_x = tf.random.normal((rule_num, att_dim,))
        model_y = tf.random.normal((rule_num, res_dim, 2))
        return Model(rule_num, att_dim, res_dim, model_x, model_y)
    raw_x = [x[(tf.argmax(y, -1) == _).numpy()] for _ in range(res_dim)]
    model_x, model_y = [], []
    idx, rng = 0, np.random.default_rng()
    while len(model_x) < rule_num:
        if len(raw_x[idx]) > 0:
            xi = raw_x[idx][rng.integers(len(raw_x[idx]))]
            yi = np.abs(rng.standard_normal((res_dim, 2,)))
            yi[:, 1] = -yi[:, 1]
            yi[idx] = -yi[idx]
            model_x.append(xi), model_y.append(yi)
        idx = (idx + 1) % res_dim
    return Model(rule_num, att_dim, res_dim, model_x, model_y)


def training(rule_num, att_dim, res_dim, x, y, bs=128, ep=500):
    s = tf.distribute.MirroredStrategy()
    ds = tf.data.Dataset.from_tensor_slices((x, y)).shuffle(1024).batch(bs).repeat(ep)
    ds = s.experimental_distribute_dataset(ds)
    with s.scope():
        # model = creat_model(rule_num, att_dim, res_dim, x, y)
        model = creat_model(rule_num, att_dim, res_dim)
        opt, tv = tf.optimizers.SGD(), model.trainable_variables


        def calculate_loss(y_true, y_pred):
            return tf.nn.compute_average_loss(
                tf.keras.losses.mse(y_true, y_pred), 
                global_batch_size=bs)

        def train_step(x, y):
            with tf.GradientTape(persistent=True) as gt:
                loss = calculate_loss(y, model.output(x))
            opt.apply_gradients(zip(gt.gradient(loss, tv), tv))
            return loss

        @tf.function
        def dist_train_step(x, y):
            per_loss = s.run(train_step, args=(x, y,))
            return s.reduce(tf.distribute.ReduceOp.SUM, per_loss, None)

        for step, (x, y) in enumerate(ds):
            loss = dist_train_step(x, y)
            tf.summary.scalar("loss", loss.numpy(), step)
    return model
    

def goss(mask):
    a = np.floor(0.3 * mask.shape[0]).astype(int)
    b = np.ceil(0.21 * mask.shape[0]).astype(int)
    return tf.concat([mask[:a], tf.random.shuffle(mask[a:])[:b]], 0).numpy()

def crossentropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred),-1)

'''
150 10 73.31
400 5  73.38
'''

def run(att_dim, res_dim, train_x, train_y, test_x, test_y):
    sub_num, rule_num, learning_rate, sub_size =10, 32, 0.5, 0.7
    sample_size = train_x.shape[0]
    sample_index = np.arange(sample_size)
    one_hot = np.eye(res_dim)[train_y].astype(np.float32)
    cce = np.zeros(sample_size)
    models, preds = [], np.zeros((sample_size ,res_dim),dtype=np.float32)
    for _ in range(sub_num):
        resi_y = one_hot - softmax(preds, -1)
        np.random.shuffle(sample_index)
        mask = sample_index[:int(sub_size * sample_size)]
        # mask = np.argsort(-train_cce)[:int(sub_size * sample_size)]
        model = training(rule_num, att_dim, res_dim, train_x[mask], resi_y[mask])
        models.append(model)
        preds += learning_rate * model.output(train_x).numpy()
        cce = crossentropy(one_hot, softmax(preds, -1))
        tf.summary.scalar('cce', np.mean(cce), _)
    test_p = tf.reduce_sum([model.output(test_x) for model in models], 0)
    return tf.math.argmax(tf.nn.softmax(test_p),-1)

    

def main():
    experiment_num, data_name = 10, 'diabetes'
    data, target, att_dim, res_dim = dataset_numeric_classification(data_name, 1)
    data = StandardScaler().fit_transform(data).astype(np.float32)
    acc = tf.metrics.Accuracy()
    for en in range(experiment_num):
        for it, (train_data, train_target, test_data, test_target) in enumerate(kfold(data, target, 10, 'numeric', random_state=en)):
            pred_target = run(att_dim, res_dim, train_data, train_target, test_data, test_target)
            acc.update_state(test_target, pred_target)
            tf.summary.scalar('acc', acc.result().numpy(), 10 * en + it)

if __name__ == "__main__":
    main()
