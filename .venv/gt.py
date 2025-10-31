##
# The MIT License (MIT)
#
# Copyright (c) 2025 Gabriel Schmidt
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

print('Generic Trainer V0.6')

print('import')
import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
import pandas as pd


inputfilename = sys.argv[1]
modelfilename = sys.argv[2]
#samplesfilename = sys.argv[3]
#resultfilename = sys.argv[4]
nepochs = int(sys.argv[3])

print('Traingsdaten bereitstellen')
xs = pd.read_csv(inputfilename, sep=',')
print('Target Daten separieren')
ys = xs.pop('ActualValue')

print('Das neuronale Netzwerk definieren und kompilieren')
model = tf.keras.Sequential([
  keras.layers.Dense(units=1, input_shape=[1])
])
model.compile(optimizer='sgd', loss='mean_squared_error')

print('Numpy arrays erzeugen')
xn = np.array(xs)
yn = np.array(ys)

print('neuronales Netzwerk trainieren')
model.fit(xn, yn, validation_split=0.1, epochs=nepochs)
print(model.summary())

print('Modell speichern')
model.save(modelfilename)

print('DONE')