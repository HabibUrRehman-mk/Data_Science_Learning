# """Dictionary to DataFrame (1)
# Pandas is an open source library, providing high-performance, easy-to-use data structures and data analysis tools for Python. Sounds promising!

# The DataFrame is one of Pandas' most important data structures. It's basically a way to store tabular data where you can label the rows and the columns. One way to build a DataFrame is from a dictionary.

# In the exercises that follow you will be working with vehicle data from different countries. Each observation corresponds to a country and the columns give information about the number of vehicles per capita, whether people drive left or right, and so on.

# Three lists are defined in the script:

# names, containing the country names for which data is available.
# dr, a list with booleans that tells whether people drive left or right in the corresponding country.
# cpc, the number of motor vehicles per 1000 people in the corresponding country.
# Each dictionary key is a column label and each value is a list which contains the column elements.

# Instructions
# 100 XP
# Import pandas as pd.
# Use the pre-defined lists to create a dictionary called my_dict. There should be three key value pairs:
# key 'country' and value names.
# key 'drives_right' and value dr.
# key 'cars_per_cap' and value cpc.
# Use pd.DataFrame() to turn your dict into a DataFrame called cars.
# Print out cars and see how beautiful it is."""


# # Pre-defined lists
# names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
# dr =  [True, False, False, False, True, True, True]
# cpc = [809, 731, 588, 18, 200, 70, 45]

# # Import pandas as pd
# import pandas as pd

# # Create dictionary my_dict with three key:value pairs: my_dict
# my_dict={'country':names,'drivers_right':dr,'cars_per_cap':cpc}

# # Build a DataFrame cars from my_dict: cars
# cars=pd.DataFrame(my_dict)

# # Print cars
# print(cars)



# """Dictionary to DataFrame (2)
# The Python code that solves the previous exercise is included in the script. Have you noticed that the row labels (i.e. the labels for the different observations) were automatically set to integers from 0 up to 6?

# To solve this a list row_labels has been created. You can use it to specify the row labels of the cars DataFrame. You do this by setting the index attribute of cars, that you can access as cars.index.

# Instructions
# 100 XP
# Hit Run Code to see that, indeed, the row labels are not correctly set.
# Specify the row labels by setting cars.index equal to row_labels.
# Print out cars again and check if the row labels are correct this time."""


# import pandas as pd

# # Build cars DataFrame
# names = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
# dr =  [True, False, False, False, True, True, True]
# cpc = [809, 731, 588, 18, 200, 70, 45]
# cars_dict = { 'country':names, 'drives_right':dr, 'cars_per_cap':cpc }
# cars = pd.DataFrame(cars_dict)
# print(cars)

# # Definition of row_labels
# row_labels = ['US', 'AUS', 'JPN', 'IN', 'RU', 'MOR', 'EG']

# # Specify row labels of cars
# cars.index=row_labels

# # Print cars again
# print(cars)


# """Square Brackets (1)
# In the video, you saw that you can index and select Pandas DataFrames in many different ways. The simplest, but not the most powerful way, is to use square brackets.

# In the sample code, the same cars data is imported from a CSV files as a Pandas DataFrame. To select only the cars_per_cap column from cars, you can use:

# cars['cars_per_cap']
# cars[['cars_per_cap']]
# The single bracket version gives a Pandas Series, the double bracket version gives a Pandas DataFrame.

# Instructions
# 100 XP
# Use single square brackets to print out the country column of cars as a Pandas Series.
# Use double square brackets to print out the country column of cars as a Pandas DataFrame.
# Use double square brackets to print out a DataFrame with both the country and drives_right columns of cars, in this order."""


# # Import cars data
# import pandas as pd
# cars = pd.read_csv('cars.csv', index_col = 0)

# # Print out country column as Pandas Series
# print(cars['country'])

# # Print out country column as Pandas DataFrame

# print(cars[['country']])
# # Print out DataFrame with country and drives_right columns
# print(cars[['country','drives_right']])





# Initialize offset
offset = -6

# Code the while loop
while offset != 0 :
    print("correcting...")
    if offset > 0 :
      offset=offset-1
    else : 
      offset +1   
    print(offset)
print('c\user')

x=1
x=x>5
print(x)
Name="EMILY"

listt=[1,2,3,4,'A','a']
print(listt.sort())

import pandas as pd