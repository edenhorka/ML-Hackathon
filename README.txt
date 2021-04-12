=================
=====README======
=================

The project contains the following files:
1. classifier.py
2. task1.py

The classifier.py file contains the functions needed for the classify function and the classifier. 
This file contains the following functions:
1. clean_data(tweets): this functions cleans the data from numbers and urls.
2. classify(tweets_array): this functions is the main function of the program. It gets one-dimentional
numpy array and returns the prediction.

The task1.py file contains all the functions which creates the classifier and trains it.
This file contains the following functions:
1. most_common(path,n): this function retruns the most common n words for each path.
2. organise_data(): this function splits the data into train and test data, and splits it to X and y.
3. get_error_rate(predictions, y): returns the difference between the predictions of the classifier
to the real tags.
4. main(d): this function combines all the different functions together.
5. find_optimal_d(): returns the optimal number of common words for each personality.
6. save_lr(lr): saves the classifier for using it in the classify function.
7. save_vec(vec): saves the vector for using it in the classify function.
