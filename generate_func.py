import os
import random as rn
import sys


BUILD_DIR = "./build/"
FUNCTIONS_FILE = BUILD_DIR + "functions.txt"

GENERATION_NUMBER = int(sys.argv[0])


# Generates random polynomial function.
# Higest degree in function could be 10
def generate_function(seed:int) -> str:
    rn.seed(seed)
    nr_of_terms = rn.randint(1,10)
    highest_degree = rn.randint(0,10)
    function = ""
    for _ in range(nr_of_terms):
        term = "+" if rn.random() < 0.5 else "-"
        term += str(rn.randint(1,10))
        if highest_degree != 0:
            term += "x^"
            term += (("" if highest_degree > 0 else "(") + str(highest_degree) + ("" if highest_degree > 0 else ")"))
        function += term
        highest_degree -= rn.randint(0, int(20/nr_of_terms))

    return function


# Create output directory if not exists.
if not os.path.isdir(BUILD_DIR):
    os.makedirs(BUILD_DIR)

with open(FUNCTIONS_FILE, mode="w") as dst_file:
    for i in range(GENERATION_NUMBER):
        function = generate_function(i)
        dst_file.write(function+"\n")
