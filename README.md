# TensorFlow Tutorial

The Tutorial includes a test program and example code for training a neural network and using it later in an application.

Try running `test.py` first. Once the program is running, TensorFlow and Keras are correctly installed.

Then use `gt.py`, the generic trainer, to read the training data from a CSV file, train the neural network, and save it as a file. It takes three parameters:
1. path and filename of CSV containing the training data (URL-Format)
2. path and filename under which the trained model shall be saved
3. number of training runs

Example:

`file://localhost/home/gabriel/PycharmProjects/TensorFlowSandbox/testdata/trainingdata.csv /home/gabriel/PycharmProjects/TensorFlowSandbox/testdata/testmodel.keras 200`

A `trainingdata.csv` is provided with this project. Please note that the last column must be titled with the keyword "ActualValue".

The trained neural network can then be applied to test samples using `gp.py` to predict suitable values. It takes three parameters:
1. path and filename of the previously trained model
2. path and filename of CSV containing the test samples (URL-Format)
3. path and filename under which the results shall be saved

Example:

`/home/gabriel/PycharmProjects/TensorFlowSandbox/testdata/testmodel.keras file://localhost/home/gabriel/PycharmProjects/TensorFlowSandbox/testdata/testsamples.csv /home/gabriel/PycharmProjects/TensorFlowSandbox/testdata/testresult.csv`

A `testsamples.csv` is provided with this project. Please note that the first column must be titled with the keyword "ID".

If everything works, you will receive a CSV file as result with three columns:
1. numerical index of the result rows
2. ID (as specified in the samples)
3. predicted result value for the data set with the corresponding ID

Congratulations! You have successfully launched your first neural network using TensorFlow and Keras!

You can now begin applying the sample programs to your data and adapting them to your requirements. You can also find further information in my [blog article.](https://www.hardwarespielerei.de/K%C3%BCnstliche%20Intelligenz/TensorFlow%20und%20Keras%20Tutorial.html)
