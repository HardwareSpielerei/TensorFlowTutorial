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

print('Generic Predicter V0.6')

print('import')
import sys
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd

modelfilename = sys.argv[1]
samplesfilename = sys.argv[2]
resultfilename = sys.argv[3]


print('Modell laden')
model = keras.models.load_model(modelfilename)
print(model.summary())

print('Input Samples laden')
samples = pd.read_csv(samplesfilename, sep=',')

print('Output vorbereiten')
output = samples.pop('ID')
output = pd.DataFrame([output]).transpose()

print('Modell anwenden')
samples = np.array(samples)
res = model.predict(samples)
output.insert(1, "Estimate", res)

print('Ergebnis speichern')
output.to_csv(resultfilename, sep=',')

print('DONE')