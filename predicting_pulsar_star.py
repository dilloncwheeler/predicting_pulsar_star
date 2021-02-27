# sounds they have heard are a pulsar star or not using machine learning. In total there are 17,898 readings 
# and of those only 1,639 of them are positively identified as a pulsar star instead of noise. 
# By: Dillon Wheeler
# 7/28/2019

import os
import pandas as pd
import seaborn as sn
import tensorflow_estimator as tfe
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, mean_squared_error


def main():
    print("Main Function")
    # Calls the machine learning function of code.
    machine_learning_function()


def load_and_trim_data(csv_file):
    """
    This function will load the data. The print statements in it will show basic information about the data that can
    be helpful for determining on what needs trimmed. The main useful ones for this data is the .info and .corr functions
    Based off of those values I dropped 4 of the 8 columns because they have a negative corrilation to the target value.
    Also in this function I replaced the spaces in the headers with underscores to help with easier programming.
    """
    pulsar_star = pd.read_csv(csv_file)

    # print(pulsar_star)  # This will print the data before trimming any data
    # print(pulsar_star.head().T)
    # print(pulsar_star.info())
    # print(pulsar_star.count().sort_values())  # This will print the number of times that value is used.
    # print(pulsar_star.describe().T)
    pulsar_star.columns = pulsar_star.columns.str.replace(' ', '_')
    # print(pulsar_star.corr()[['target_class']].sort_values('target_class'))

    pulsar_star = pulsar_star.drop(['_Mean_of_the_integrated_profile', '_Excess_kurtosis_of_the_DM-SNR_curve',
                                    '_Standard_deviation_of_the_integrated_profile', '_Skewness_of_the_DM-SNR_curve'],
                                   axis=1)

    # print(pulsar_star.corr()[['target_class']].sort_values('target_class'))
    return pulsar_star


def machine_learning_function():
    """
    This is the function that does the most in this script. It is going to take in the csv and then run it though the
    load and trim data function. We then setup our x and y values. (This is after the data has been trimmed.)
    We then split the data into train test. This is the typical 70/30 split and set up scalers for the data. 
    
    After that we setup our feature columns using tensorflow and we also setup an input function. 
    The input function is then passed into a DNNClassier function from tensorflow. The test portion of the 
    data is then run though the model and checked for accuracty which prints out the confusion metrics along
    with a graph showing the results. The average of this code has been around 98% accurate. 
    """
    df = load_and_trim_data("../input/pulsar_stars.csv")

    X_data = df.drop(['target_class'], axis=1)
    y_data = df['target_class']

    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.3)
    scaler = MinMaxScaler()
    scaler.fit(X_train)

    X_train = pd.DataFrame(data=scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(data=scaler.fit_transform(X_test), columns=X_test.columns, index=X_test.index)

    dm_snr_mean = tf.feature_column.numeric_column('_Mean_of_the_DM-SNR_curve')
    dm_snr_std_dev = tf.feature_column.numeric_column('_Standard_deviation_of_the_DM-SNR_curve')
    dm_snr_skewness = tf.feature_column.numeric_column('_Skewness_of_the_integrated_profile')
    excess_kurtosis = tf.feature_column.numeric_column('_Excess_kurtosis_of_the_integrated_profile')

    feature_columns = [dm_snr_mean, dm_snr_std_dev, dm_snr_skewness, excess_kurtosis]
    input_function = tfe.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=100,
                                                          shuffle=True)

    model = tfe.estimator.DNNClassifier(hidden_units=[4, 4, 4], feature_columns=feature_columns)
    model.train(input_fn=input_function)

    pred_fn = tfe.estimator.inputs.pandas_input_fn(x=X_test, batch_size=len(y_test), shuffle=False)
    note_predictions = list(model.predict(input_fn=pred_fn))
    # print(note_predictions)
    final_preds = []
    for pred in note_predictions:
        final_preds.append(pred['class_ids'][0])
    # print(final_preds)
    print(confusion_matrix(y_test, final_preds))
    accuracy = accuracy_score(y_test, final_preds)
    title = ("Accuracy of Machine Learning \n" + '{}'.format(accuracy))
    print(classification_report(y_test, final_preds))
    plt_creation(confusion_matrix(y_test, final_preds), title)
    print(mean_squared_error(y_test, final_preds))
    plt.show()
    print(accuracy)


def plt_creation(input_confusion_matrix, graph_title):
    """
    This is the function that we are calling to make the heat maps for each of our functions. I made it take in the
    matrix from the function and then I made it take in the graph title as well.
    """
    sn.set(font_scale=1)
    sn.heatmap(input_confusion_matrix, annot=True, annot_kws={"size": 16})
    plt.title(graph_title)
    plt.show()


if __name__ == '__main__':
    main()
