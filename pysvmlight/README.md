PySVMLight
==========

A Python binding to the [SVM-Light](http://svmlight.joachims.org/) support vector machine library by Thorsten Joachims.

Written by Bill Cauchois (<wcauchois@gmail.com>), with thanks to Lucas Beyer and n0mad for their contributions.

Installation
------------
PySVMLight uses distutils for setup. Installation is as simple as

    $ chmod +x setup.py
    $ ./setup.py --help
    $ ./setup.py build

If you want to install SVMLight to your PYTHONPATH, type:

    $ ./setup.py install

(You may need to execute this command as the superuser.) Otherwise, look in the build/ directory to find svmlight.so and copy that file to the directory of your project. You should now be able to `import svmlight`.

Getting Started
---------------
See examples/simple.py for example usage.

Reference
---------

If you type `help(svmlight)`, you will see that there are currently three functions.

    learn(training_data, **options) -> model

Train a model based on a set of training data. The training data should be in the following format:

    >> (<label>, [(<feature>, <value>), ...])

or

    >> (<label>, [(<feature>, <value>), ...], <queryid>)

See examples/data.py for an example of some training data. Available options include (corresponding roughly to the command-line options for `svmlight` detailed on [this page](http://svmlight.joachims.org/) under the section titled "How to use"):

 - `type`: select between 'classification', 'regression', 'ranking' (preference ranking), and 'optimization'.
 - `kernel`: select between 'linear', 'polynomial', 'rbf', and 'sigmoid'.
 - `verbosity`: set the verbosity level (default 0).
 - `C`: trade-off between training error and margin.
 - `poly_degree`: parameter d in polynomial kernel.
 - `rbf_gamma`: parameter gamma in rbf kernel.
 - `coef_lin`
 - `coef_const`

The result of this call is a model that you can pass to classify().

    classify(model, test_data, **options) -> predictions

Classify a set of test data using the provided model. The test data should be in the same format as training data (see above). The result will be a list of floats, corresponding to predicted labels for each of the test instances.

    write_model(model, filename) -> None

Write the provided model to the specified file. The file format used is the same format as that used by the command-line `svmlight` program.

    read_model(filename) -> model

Read a model that was saved using write_model().

