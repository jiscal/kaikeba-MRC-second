import tensorflow as tf

class C2QAttention(tf.keras.layers.Layer):

    def call(self, similarity, qencode):
        #100 100维数据
        # 1. 对qecncode进行扩展维度 ：tf.expand_dims、
        #None,1,100,100
        qencode=tf.expand_dims(qencode,1)

        # 2. softmax函数处理相似度矩阵：tf.keras.activations.softmax
        #100 100数据
        similarity=tf.keras.activations.softmax(similarity)
        # 3. 对处理结果扩展维度：tf.expand_dims
        #None,100,100,1
        similarity=tf.expand_dims(similarity,-1)
        # 4. 加权求和：tf.math.reduce_sum
        c2q_att=tf.multiply(similarity,qencode)
        c2q_att=tf.math.reduce_sum(c2q_att,axis=2)
        return c2q_att

class Q2CAttention(tf.keras.layers.Layer):

    def call(self, similarity, cencode):

        # 1.计算similarity矩阵最大值：tf.math.reduce_max
        #100
        similarity= tf.math.reduce_max(similarity,axis=-1)
        # 2.使用 softmax函数处理最大值的相似度矩阵：tf.keras.activations.softmax
        similarity=tf.keras.activations.softmax(similarity)

        # 3.维度处理：tf.expand_dims
        #100,1
        similarity=tf.expand_dims(similarity,-1)
        # 4.加权求和：tf.math.reduce_sum
        q2c_att = tf.multiply(similarity, cencode)
        #100
        q2c_att=tf.math.reduce_sum(cencode,axis=2)

        # 5.再次维度处理加权求和后的结果：tf.expand_dims
        q2c_att=tf.expand_dims(q2c_att,1)
        # 6.获取重复的次数： cencode.shape[1]
        nums=cencode.shape[1]
        # 7.重复拼接获取最终矩阵：tf.tile
        q2c_att=tf.tile(q2c_att,[1,nums,1])

        return q2c_att
