{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 25,
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn import datasets\n",
    "import seaborn as sns\n",
    "import numpy as np\n",
    "from sklearn import tree\n",
    "from sklearn.metrics import accuracy_score\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris = datasets.load_iris()\n",
    "digits = datasets.load_digits()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[ 0.  0.  5. ...  0.  0.  0.]\n",
      " [ 0.  0.  0. ... 10.  0.  0.]\n",
      " [ 0.  0.  0. ... 16.  9.  0.]\n",
      " ...\n",
      " [ 0.  0.  1. ...  6.  0.  0.]\n",
      " [ 0.  0.  2. ... 12.  0.  0.]\n",
      " [ 0.  0. 10. ... 12.  1.  0.]]\n"
     ]
    }
   ],
   "source": [
    "print(digits.data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 1 2 ... 8 9 8]\n"
     ]
    }
   ],
   "source": [
    "print(digits.target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[5.1 3.5 1.4 0.2]\n",
      " [4.9 3.  1.4 0.2]\n",
      " [4.7 3.2 1.3 0.2]\n",
      " [4.6 3.1 1.5 0.2]\n",
      " [5.  3.6 1.4 0.2]\n",
      " [5.4 3.9 1.7 0.4]\n",
      " [4.6 3.4 1.4 0.3]\n",
      " [5.  3.4 1.5 0.2]\n",
      " [4.4 2.9 1.4 0.2]\n",
      " [4.9 3.1 1.5 0.1]\n",
      " [5.4 3.7 1.5 0.2]\n",
      " [4.8 3.4 1.6 0.2]\n",
      " [4.8 3.  1.4 0.1]\n",
      " [4.3 3.  1.1 0.1]\n",
      " [5.8 4.  1.2 0.2]\n",
      " [5.7 4.4 1.5 0.4]\n",
      " [5.4 3.9 1.3 0.4]\n",
      " [5.1 3.5 1.4 0.3]\n",
      " [5.7 3.8 1.7 0.3]\n",
      " [5.1 3.8 1.5 0.3]\n",
      " [5.4 3.4 1.7 0.2]\n",
      " [5.1 3.7 1.5 0.4]\n",
      " [4.6 3.6 1.  0.2]\n",
      " [5.1 3.3 1.7 0.5]\n",
      " [4.8 3.4 1.9 0.2]\n",
      " [5.  3.  1.6 0.2]\n",
      " [5.  3.4 1.6 0.4]\n",
      " [5.2 3.5 1.5 0.2]\n",
      " [5.2 3.4 1.4 0.2]\n",
      " [4.7 3.2 1.6 0.2]\n",
      " [4.8 3.1 1.6 0.2]\n",
      " [5.4 3.4 1.5 0.4]\n",
      " [5.2 4.1 1.5 0.1]\n",
      " [5.5 4.2 1.4 0.2]\n",
      " [4.9 3.1 1.5 0.1]\n",
      " [5.  3.2 1.2 0.2]\n",
      " [5.5 3.5 1.3 0.2]\n",
      " [4.9 3.1 1.5 0.1]\n",
      " [4.4 3.  1.3 0.2]\n",
      " [5.1 3.4 1.5 0.2]\n",
      " [5.  3.5 1.3 0.3]\n",
      " [4.5 2.3 1.3 0.3]\n",
      " [4.4 3.2 1.3 0.2]\n",
      " [5.  3.5 1.6 0.6]\n",
      " [5.1 3.8 1.9 0.4]\n",
      " [4.8 3.  1.4 0.3]\n",
      " [5.1 3.8 1.6 0.2]\n",
      " [4.6 3.2 1.4 0.2]\n",
      " [5.3 3.7 1.5 0.2]\n",
      " [5.  3.3 1.4 0.2]\n",
      " [7.  3.2 4.7 1.4]\n",
      " [6.4 3.2 4.5 1.5]\n",
      " [6.9 3.1 4.9 1.5]\n",
      " [5.5 2.3 4.  1.3]\n",
      " [6.5 2.8 4.6 1.5]\n",
      " [5.7 2.8 4.5 1.3]\n",
      " [6.3 3.3 4.7 1.6]\n",
      " [4.9 2.4 3.3 1. ]\n",
      " [6.6 2.9 4.6 1.3]\n",
      " [5.2 2.7 3.9 1.4]\n",
      " [5.  2.  3.5 1. ]\n",
      " [5.9 3.  4.2 1.5]\n",
      " [6.  2.2 4.  1. ]\n",
      " [6.1 2.9 4.7 1.4]\n",
      " [5.6 2.9 3.6 1.3]\n",
      " [6.7 3.1 4.4 1.4]\n",
      " [5.6 3.  4.5 1.5]\n",
      " [5.8 2.7 4.1 1. ]\n",
      " [6.2 2.2 4.5 1.5]\n",
      " [5.6 2.5 3.9 1.1]\n",
      " [5.9 3.2 4.8 1.8]\n",
      " [6.1 2.8 4.  1.3]\n",
      " [6.3 2.5 4.9 1.5]\n",
      " [6.1 2.8 4.7 1.2]\n",
      " [6.4 2.9 4.3 1.3]\n",
      " [6.6 3.  4.4 1.4]\n",
      " [6.8 2.8 4.8 1.4]\n",
      " [6.7 3.  5.  1.7]\n",
      " [6.  2.9 4.5 1.5]\n",
      " [5.7 2.6 3.5 1. ]\n",
      " [5.5 2.4 3.8 1.1]\n",
      " [5.5 2.4 3.7 1. ]\n",
      " [5.8 2.7 3.9 1.2]\n",
      " [6.  2.7 5.1 1.6]\n",
      " [5.4 3.  4.5 1.5]\n",
      " [6.  3.4 4.5 1.6]\n",
      " [6.7 3.1 4.7 1.5]\n",
      " [6.3 2.3 4.4 1.3]\n",
      " [5.6 3.  4.1 1.3]\n",
      " [5.5 2.5 4.  1.3]\n",
      " [5.5 2.6 4.4 1.2]\n",
      " [6.1 3.  4.6 1.4]\n",
      " [5.8 2.6 4.  1.2]\n",
      " [5.  2.3 3.3 1. ]\n",
      " [5.6 2.7 4.2 1.3]\n",
      " [5.7 3.  4.2 1.2]\n",
      " [5.7 2.9 4.2 1.3]\n",
      " [6.2 2.9 4.3 1.3]\n",
      " [5.1 2.5 3.  1.1]\n",
      " [5.7 2.8 4.1 1.3]\n",
      " [6.3 3.3 6.  2.5]\n",
      " [5.8 2.7 5.1 1.9]\n",
      " [7.1 3.  5.9 2.1]\n",
      " [6.3 2.9 5.6 1.8]\n",
      " [6.5 3.  5.8 2.2]\n",
      " [7.6 3.  6.6 2.1]\n",
      " [4.9 2.5 4.5 1.7]\n",
      " [7.3 2.9 6.3 1.8]\n",
      " [6.7 2.5 5.8 1.8]\n",
      " [7.2 3.6 6.1 2.5]\n",
      " [6.5 3.2 5.1 2. ]\n",
      " [6.4 2.7 5.3 1.9]\n",
      " [6.8 3.  5.5 2.1]\n",
      " [5.7 2.5 5.  2. ]\n",
      " [5.8 2.8 5.1 2.4]\n",
      " [6.4 3.2 5.3 2.3]\n",
      " [6.5 3.  5.5 1.8]\n",
      " [7.7 3.8 6.7 2.2]\n",
      " [7.7 2.6 6.9 2.3]\n",
      " [6.  2.2 5.  1.5]\n",
      " [6.9 3.2 5.7 2.3]\n",
      " [5.6 2.8 4.9 2. ]\n",
      " [7.7 2.8 6.7 2. ]\n",
      " [6.3 2.7 4.9 1.8]\n",
      " [6.7 3.3 5.7 2.1]\n",
      " [7.2 3.2 6.  1.8]\n",
      " [6.2 2.8 4.8 1.8]\n",
      " [6.1 3.  4.9 1.8]\n",
      " [6.4 2.8 5.6 2.1]\n",
      " [7.2 3.  5.8 1.6]\n",
      " [7.4 2.8 6.1 1.9]\n",
      " [7.9 3.8 6.4 2. ]\n",
      " [6.4 2.8 5.6 2.2]\n",
      " [6.3 2.8 5.1 1.5]\n",
      " [6.1 2.6 5.6 1.4]\n",
      " [7.7 3.  6.1 2.3]\n",
      " [6.3 3.4 5.6 2.4]\n",
      " [6.4 3.1 5.5 1.8]\n",
      " [6.  3.  4.8 1.8]\n",
      " [6.9 3.1 5.4 2.1]\n",
      " [6.7 3.1 5.6 2.4]\n",
      " [6.9 3.1 5.1 2.3]\n",
      " [5.8 2.7 5.1 1.9]\n",
      " [6.8 3.2 5.9 2.3]\n",
      " [6.7 3.3 5.7 2.5]\n",
      " [6.7 3.  5.2 2.3]\n",
      " [6.3 2.5 5.  1.9]\n",
      " [6.5 3.  5.2 2. ]\n",
      " [6.2 3.4 5.4 2.3]\n",
      " [5.9 3.  5.1 1.8]]\n"
     ]
    }
   ],
   "source": [
    "print(iris.data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0\n",
      " 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\n",
      " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2\n",
      " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2\n",
      " 2 2]\n"
     ]
    }
   ],
   "source": [
    "print(iris.target)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['setosa' 'versicolor' 'virginica']\n"
     ]
    }
   ],
   "source": [
    "print(iris.target_names)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris_data = iris.data #variable for array of the data\n",
    "iris_target = iris.target #variable for array of the labels\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 72x720 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "import matplotlib.pyplot as plt\n",
    "plt.show()\n",
    "sns.boxplot(data = iris_data,width=0.5,fliersize=5)\n",
    "sns.set(rc={'figure.figsize':(1,10)})\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris_test_ids = np.random.permutation(len(iris_data)) #randomly splitting the data set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [],
   "source": [
    "#splitting and leaving last 15 entries for testing, rest for training\n",
    "iris_train_one = iris_data[iris_test_ids[:-15]]\n",
    "iris_test_one = iris_data[iris_test_ids[-15:]]\n",
    "iris_train_two = iris_target[iris_test_ids[:-15]]\n",
    "iris_test_two = iris_target[iris_test_ids[-15:]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [],
   "source": [
    "iris_classify = tree.DecisionTreeClassifier()#using the decision tree for classification\n",
    "iris_classify.fit(iris_train_one, iris_train_two) #training or fitting the classifier using the training set\n",
    "iris_predict = iris_classify.predict(iris_test_one) #making predictions on the test dataset\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[0 2 1 2 1 2 2 1 1 1 0 0 0 0 0]\n",
      "[0 2 1 2 1 2 2 1 1 1 0 0 0 0 0]\n",
      "100.0\n"
     ]
    }
   ],
   "source": [
    "print(iris_predict) #labels predicted (flower species)\n",
    "print (iris_test_two) #actual labels\n",
    "print (accuracy_score(iris_predict, iris_test_two)*100) #accuracy metric"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
