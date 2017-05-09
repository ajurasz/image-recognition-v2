import os
import numpy as np
import tensorflow as tf

cat_labels = [8, 10, 11, 55, 95, 174]


def predict(app, image):
    app_root = os.path.sep.join(app.instance_path.split(os.path.sep)[:-1])
    with tf.gfile.FastGFile(os.path.join(
            app_root, 'ml', 'model', 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

    with tf.Session() as sess:
        softmax_tensor = sess.graph.get_tensor_by_name('softmax:0')
        predictions = sess.run(softmax_tensor,
                               {'DecodeJpeg/contents:0': image})
        predictions = np.squeeze(predictions)
        best = predictions.argsort()[-1:][::-1]
        return best in cat_labels
