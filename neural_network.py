import tensorflow as tf

def loss_mse(target_y, predicted_y):
  return tf.reduce_mean(tf.square(target_y - predicted_y))

def loss_cross_entropy(target_y,predicted_y):
  return -tf.reduce_mean(tf.reduce_sum(target_y * tf.math.log(predicted_y + 1e-12),axis=0))

class Layer(tf.Module):
  def __init__(self,input_size,output_size,is_last=False, **kwargs):
    super().__init__(**kwargs)
    self.w = tf.Variable(tf.random.normal([input_size, output_size]) * tf.sqrt(2 / (input_size + output_size)), name='w')
    self.b = tf.Variable(0.0,name='b')
    self.is_last = is_last
    self.input_size = input_size
    self.output_size = output_size

  def __call__(self, x):
    if self.is_last:
      result = tf.nn.relu(x @ self.w + self.b)
    else:
      result = tf.nn.relu(x @ self.w + self.b)
    return result


class NeuralNetwork(tf.Module):
  def __init__(self,layers,**kwargs):
    super().__init__(**kwargs)

    # Check if layers sizes are inconsistent
    first_layer = True
    for layer in layers:
      if first_layer:
        first_layer = False
      else:
        if layer.input_size != previous_output:
          print('Inconsistent layers')
      previous_output = layer.output_size
    if layers[-1].is_last == False:
      print('Last layer is_last = True!')
    self._layers = layers

  def __call__(self, x0):
    for i_layer in self._layers:
      x0 = i_layer(x0)
    return x0

def train_nn(neural_network,x,y,learning_rate):

  with tf.GradientTape(persistent=True) as t:
    current_loss = loss_mse(y, neural_network(x))

  gradiente = t.gradient(current_loss, neural_network.trainable_variables)

  for i_trainable_variable,i_gradient in zip(neural_network.trainable_variables,gradiente):
    i_trainable_variable.assign_sub(learning_rate*i_gradient)