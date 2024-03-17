#Generate random text data
from random import randint, choice
from string import ascii_letters, digits, punctuation, whitespace
import json

# Example dictionary
characters = digits + ascii_letters
my_dict = {'PoolSize': len(characters), 'pool': characters}
file_path = 'DataDetails.json'
print(sorted(set(characters)))
print(characters)
# Specify the file path

# Write the dictionary to a file
with open(file_path, 'w') as file:
    json.dump(my_dict, file)

print("Pool Size:", len(characters))
size = int(input("Enter the size of data: "))
with open('testfile.txt', 'w') as file:
    for i in range(size):
        file.write(str(choice(characters)))
    file.close()
