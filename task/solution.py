import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

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

    # Print the distribution of 'Zip_loc' values in the training set as a dictionary
    print(X_train['Zip_loc'].value_counts().to_dict())
