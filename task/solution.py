import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from category_encoders import TargetEncoder
from sklearn.metrics import classification_report

if __name__ == '__main__':
    if not os.path.exists('../Data'):
        os.mkdir('../Data')

    # Download data if it is unavailable.
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")

    # Load the house dataset from a CSV file
    house_df = pd.read_csv('../Data/house_class.csv')

    # Print the number of rows in the dataset
    # print(len(house_df))

    # Print the number of columns in the dataset
    # print(house_df.columns.size)

    # Check if there are any null values in the dataset
    # print(house_df.isnull().values.any())

    # Print the maximum number of rooms in the 'Room' column
    # print(house_df['Room'].max())

    # Print the mean area of the houses in the 'Area' column
    # print(house_df['Area'].mean())

    # Print the number of unique values in the 'Zip_loc' column
    # print(house_df['Zip_loc'].unique().size)

    # Define the feature matrix X by selecting specific columns from the DataFrame
    X = house_df.loc[:, ['Area', 'Room', 'Lon', 'Lat', 'Zip_area', 'Zip_loc']]

    # Define the target vector y by selecting the 'Price' column from the DataFrame
    y = house_df.loc[:, 'Price']

    # Split the dataset into training and testing sets
    # The test set will be 30% of the data
    # Use 'Zip_loc' column for stratification to ensure the same distribution of values in both sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1,
                                                        stratify=X['Zip_loc'].values)

    # Initialize the OneHotEncoder with drop='first' to drop the first category of each categorical feature
    enc = OneHotEncoder(drop='first')
    ord = OrdinalEncoder()
    target_encoder = TargetEncoder(cols=['Room', 'Zip_area', 'Zip_loc'])

    # Fit the encoder on the training data's categorical features: Zip_area, Zip_loc, and Room
    enc.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
    ord.fit(X_train[['Zip_area', 'Zip_loc', 'Room']])
    target_encoder.fit(X_train[[ 'Room', 'Zip_area', 'Zip_loc']], y_train)

    # Transform the categorical features of the training data into one-hot encoded arrays
    # Convert the resulting NumPy array to a DataFrame, setting the index to match the original training data
    # Add a prefix 'enc' to the column names to indicate these are encoded features
    X_train_transformed_enc = pd.DataFrame(enc.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                       index=X_train.index).add_prefix('enc')
    X_train_transformed_ord = pd.DataFrame(ord.transform(X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                       index=X_train.index).add_prefix('enc')
    X_train_transformed_targ = target_encoder.transform(X_train[['Room', 'Zip_area', 'Zip_loc']])

    # Transform the categorical features of the test data into one-hot encoded arrays
    # Convert the resulting NumPy array to a DataFrame, setting the index to match the original test data
    # Add a prefix 'enc' to the column names to indicate these are encoded features
    X_test_transformed_enc = pd.DataFrame(enc.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                      index=X_test.index).add_prefix('enc')
    X_test_transformed_ord = pd.DataFrame(ord.transform(X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                      index=X_test.index).add_prefix('enc')
    X_test_transformed_targ = target_encoder.transform(X_test[['Room', 'Zip_area', 'Zip_loc']])

    # Join the transformed (encoded) training data with the original numerical features: Area, Lon, and Lat
    X_train_final_enc = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed_enc)
    X_train_final_ord = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed_ord)
    X_train_final_targ = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed_targ)

    # Join the transformed (encoded) test data with the original numerical features: Area, Lon, and Lat
    X_test_final_enc = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed_enc)
    X_test_final_ord = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed_ord)
    X_test_final_targ = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed_targ)

    # Initialize the DecisionTreeClassifier with specified parameters
    clf = DecisionTreeClassifier(criterion='entropy', max_features=3, splitter='best', max_depth=6, min_samples_split=4,
                                 random_state=3)


    # Fit the model to the final training data and corresponding labels
    clf.fit(X_train_final_enc, y_train)

    # Predict the labels for the final test data
    y_predict_enc = clf.predict(X_test_final_enc)

    enc_precision = classification_report(y_test, y_predict_enc, output_dict=True)

    print(f"OneHotEncoder:{enc_precision['macro avg']['f1-score']:.2f}")
    # Fit the model to the final training data and corresponding labels
    clf.fit(X_train_final_ord, y_train)

    # Predict the labels for the final test data
    y_predict_ord = clf.predict(X_test_final_ord)

    ord_precision = classification_report(y_test, y_predict_ord, output_dict=True)
    print(f"OrdinalEncoder:{ord_precision['macro avg']['f1-score']:.2f}")

    # Fit the model to the final training data and corresponding labels
    clf.fit(X_train_final_targ, y_train)

    # Predict the labels for the final test data
    y_predict_targ = clf.predict(X_test_final_targ)

    targ_precision = classification_report(y_test, y_predict_targ, output_dict=True)
    print(f"TargetEncoder:{targ_precision['macro avg']['f1-score']:.2f}")

    # Calculate and print the accuracy score of the model on the test data
    # print(accuracy_score(y_test, y_predict))

    # Print the distribution of 'Zip_loc' values in the training set as a dictionary
    # print(X_train['Zip_loc'].value_counts().to_dict())


