### MNIST Data

MNIST Data in CSV format is available from [here](https://pjreddie.com/projects/mnist-in-csv/).

    wget https://pjreddie.com/media/files/mnist_train.csv
    wget https://pjreddie.com/media/files/mnist_test.csv

Running the above commands results in a mnist\_train.csv and mnist\_test.csv files in the current directory. Format of the file is label (number from 0-9), followed by 28x28 (784 comma separated numbers from 0-127, representing black and white intensity values for each pixel). Training set has 60k records and test set has 10k records.

### Boston House Prices

Needed only for the Tensorflow Linear Regression example, available from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/housing). I manually navigated to the URL, and copy-pasted the file directly to data/housing.data.

