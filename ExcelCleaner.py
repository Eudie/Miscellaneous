#!/usr/bin/python
# Author: Eudie
# This code is to delete special character from the cell of excel file. It can work on file as well as directory.

import os
import sys
import pandas as pd


def file_cleaner(file_to_read):
    df = pd.read_excel(file_to_read)
    df = df.replace({'\n': ' ', ',': ' ', '"': ' ', '\t': ' ', '[|]': ' '}, regex=True)
    print df
    writer = pd.ExcelWriter(os.path.splitext(file_to_read)[0] + '_cleaned.xlsx')
    df.to_excel(writer, sheet_name='Sheet1', index=False)


def main():
    flag_name = sys.argv[1]
    if os.path.splitext(flag_name)[1] == ".xlsx":
        file_cleaner(flag_name)
    else:
        list_of_files = os.listdir(flag_name)
        print list_of_files
        for individual_file in list_of_files:
            if os.path.splitext(individual_file)[1] == ".xlsx":
                file_cleaner(flag_name + "/" + individual_file)

if __name__ == "__main__":
    main()
