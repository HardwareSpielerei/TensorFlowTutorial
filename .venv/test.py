# MIT License
#
# Copyright (c) 2025 Gabriel Schmidt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

print('import')
import tensorflow as tf
import numpy as np
from tensorflow import keras

print(tf.__version__)

print('Das neuronale Netzwerk definieren und kompilieren')
model = tf.keras.Sequential([
  keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

print('Daten bereitstellen')
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
print(xs.dtype)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)
print(ys.dtype)

print('neuronales Netzwerk trainieren')
model.fit(xs, ys, validation_split=0.1, epochs=200)

print('Modell verwenden')
test_input = np.array([-100, -5, -1, 3, 5, 10.0, 100.0], dtype=float)
print(test_input)
result = model.predict(test_input)
print(result)

print('DONE')
