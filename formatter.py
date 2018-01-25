import os
""" Formats image name from 2018-01-... """
i = 1
dir_list = list(map(str, range(1,7)))
print (dir_list)
for directory in os.listdir("."):
    print(directory)
    if directory in dir_list:
        for filename in os.listdir(directory):
            if filename.startswith('2018-01'):
                temp_name = str(i) + '.jpg'
                print(filename, temp_name)
                os.rename(os.path.join(directory, filename),os.path.join(directory, temp_name))
                i += 1
    i = 1