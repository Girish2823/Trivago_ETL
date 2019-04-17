# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 11:59:07 2018

@author: giris
"""

import csv
import sqlite3
import os.path

db_filename = "E:\Lending\loan_data.db"

fname  = 'E:\clicklog_2017-03-20.csv'
check = os.path.isfile(fname)

if check == TRUE :
    
    try:
        conn = sqlite3.connect(db_filename)
        c = conn.cursor()
    except Error as e:
        print(e)

    sql = """
    INSERT INTO Triv (User_id,time,action,destination,hotel)
    values (:User_id,:time,:action,:destination,:hotel)
    """

    #Dropping the existing Table
    stmt = 'DROP TABLE Triv'
    conn.execute(stmt)    
    #Create a Table To insert the data from the csv file
    statement = 'create table Triv(User_id text,time text,action text,destination text,hotel text)'
    conn.execute(statement)
    print("Table created")

    #reading the File and inserting data in database
    with open('E:\clicklog_2017-03-20.csv',encoding = 'UTF-8') as csvDataFile:
        csvReader = csv.reader(csvDataFile)
        c.executemany(sql,csvReader)
        

    conn.commit()
    conn.close()
    print("Insertion Completed")

else:
    print("File not present)


    