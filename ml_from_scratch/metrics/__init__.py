#Import the accuracy_score classes from the _classification module
from ._classification import accuracy_score

#List of names to be exported when using the import statement
__all__ = {
    "accuracy" : accuracy_score #Expose accuracy_score class
}