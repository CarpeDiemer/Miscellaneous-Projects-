import tensorflow as tf

tf.reset_default_graph()
g = tf.Graph()
#使用vgg作为卷积模型
net = vgg16.get_vgg_model()
with tf.Session(graph=g) as sess:
    #此处我们使用以content_img为基的正态分布作为初始输入
    #我们同样可以选择其他初始化方式，例如直接输入原图content_img或
    #风格图style_img
    net_input = tf.get_variable(
       name='input',
       shape=content_img.shape,
       dtype=tf.float32,
       initializer=tf.random_normal_initializer(
           mean=np.mean(content_img), stddev=np.std(content_img)))
    
    #将网络定义读入当下的图g中，同时声明网络的输入层为net_input
    tf.import_graph_def(
        net['graph_def'],
        name='net',
        input_map={'images:0': net_input})

 #我们拿某一层的输出出来
content_layer = 'net/conv3_2/conv3_2:0'

#获得content值输出的ac
with tf.Session(graph=g) as sess:
    content_features = g.get_tensor_by_name(content_layer).eval(
            session=sess,
            feed_dict={x: content_img,
            #此处因为并不需要规约权值，故将dropout关闭
                    'net/dropout_1/random_uniform:0': keep_probability,
                    'net/dropout/random_uniform:0': keep_probability})

#获得MSE目标函数
with tf.Session(graph=g) as sess:
    content_loss = tf.nn.l2_loss((g.get_tensor_by_name(content_layer) -
                                 content_features) /
                                 content_features.size)

#挑选出不同的网络层
style_layers = ['net/conv1_1/conv1_1:0',
                'net/conv2_1/conv2_1:0',
                'net/conv3_1/conv3_1:0',
                'net/conv4_1/conv4_1:0',
                'net/conv5_1/conv5_1:0']
style_activations = []

#为Style图片求激活函数值
with tf.Session(graph=g) as sess:
    for style_i in style_layers:
        style_activation_i = g.get_tensor_by_name(style_i).eval(
            feed_dict={x: style_img,
                    'net/dropout_1/random_uniform:0': keep_probability,
                    'net/dropout/random_uniform:0': keep_probability})
        style_activations.append(style_activation_i)

#为Style图片自身求gram matrix
style_features = []
for style_activation_i in style_activations:
    s_i = np.reshape(style_activation_i, [-1, style_activation_i.shape[-1]])
    gram_matrix = np.matmul(s_i.T, s_i) / s_i.size
    style_features.append(gram_matrix.astype(np.float32))

#求input的gram martix，并与Style的gram matrix求l2目标函数
with tf.Session(graph=g) as sess:
    style_loss = np.float32(0.0)
    for style_layer_i, style_gram_i in zip(style_layers, style_features):
        layer_i = g.get_tensor_by_name(style_layer_i)
        layer_shape = layer_i.get_shape().as_list()
        layer_size = layer_shape[1] * layer_shape[2] * layer_shape[3]
        layer_flat = tf.reshape(layer_i, [-1, layer_shape[3]])
        gram_matrix = tf.matmul(tf.transpose(layer_flat), layer_flat) / layer_size
        style_loss = tf.add(style_loss, tf.nn.l2_loss((gram_matrix - style_gram_i) / np.float32(style_gram_i.size)))

with tf.Session(graph=g) as sess:
    loss = 5.0 * content_loss + 1.0 * style_loss 
    optimizer = tf.train.AdamOptimizer(0.05).minimize(loss)

with tf.Session(graph=g) as sess:
    sess.run(tf.global_variables_initializer())
    n_iterations = 150
        
    for it_i in range(n_iterations):
        _, this_loss, synth = sess.run([optimizer, loss, net_input], feed_dict={
                    'net/dropout_1/random_uniform:0': keep_probability,
                    'net/dropout/random_uniform:0': keep_probability})