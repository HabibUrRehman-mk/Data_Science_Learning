import numpy as np

baseball = [180, 215, 210, 210, 188, 176, 209, 200]
np_baseball=np.array(baseball)
print(np_baseball)




"""Baseball players' height
You are a huge baseball fan. You decide to call the MLB (Major League Baseball) and ask around for 
some more statistics on the height of the main players. They pass along data on more than a thousand players, 
which is stored as a regular Python list: height_in. The height is expressed in inches. Can you make a numpy array 
out of it and convert the units to meters?
height_in is already available and the numpy package is loaded, so you can start straight away"""

# Import numpy
import numpy as np

# Create a numpy array from height_in: np_height_in
height_in = [74, 78, 76, 75, 72, 70, 73, 77, 79, 80, 75, 74, 76, 78, 72]
np_height_in=np.array(height_in)

# Print out np_height_in
print(np_height_in)
# Convert np_height_in to m: np_height_m
np_height_m=np_height_in * 0.0254

# Print np_height_m
print (np_height_m)


"""Subsetting 2D NumPy Arrays
If your 2D numpy array has a regular structure, i.e. each row and column has a fixed number of values, 
complicated ways of subsetting become very easy. Have a look at the code below where the elements "a" and "c" 
are extracted from a list of lists.
# numpy
import numpy as np
np_x = np.array(x)
np_x[:, 0]
The indexes before the comma refer to the rows, while those after the comma refer to the columns. 
The : is for slicing; in this example, it tells Python to include all rows."""

import numpy as np

np_baseball = np.array(baseball)

# Print out the 50th row of np_baseball

print(np_baseball[49,:])
# Select the entire second column of np_baseball: np_weight_lb
np_weight_lb=np_baseball[:,1]

# Print out height of 124th player
print(np_baseball[123,0])



"""2D Arithmetic
2D numpy arrays can perform calculations element by element, like numpy arrays.

np_baseball is coded for you; it's again a 2D numpy array with 3 columns representing height (in inches), 
weight (in pounds) and age (in years). baseball is available as a regular list of lists and updated is available as 
2D numpy array.
Instructions
100 XP
You managed to get hold of the changes in height, weight and age of all baseball players. 
It is available as a 2D numpy array, updated. Add np_baseball and updated and print out the result.
You want to convert the units of height and weight to metric (meters and kilograms, respectively).
 As a first step, create a numpy array with three values: 0.0254, 0.453592 and 1. Name this array conversion.
Multiply np_baseball with conversion and print out the result."""


import numpy as np

np_baseball = np.array(baseball)

# Print out addition of np_baseball and updated
print(np_baseball+updated)

# Create numpy array: conversion

conversion=np.array([0.0254,0.453592,1])
# Print out product of np_baseball and conversion
print(np_baseball * conversion)