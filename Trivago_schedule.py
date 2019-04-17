import csv
import sqlite3
import os.path


db_filename = "E:\Lending\loan_data.db"

#The path of the file that is to be transferred to the database. We, check the presence of file, if it is not present then the message "File not present is generated" else the 
#process is continued to insert the csv file data into the database. I have used a path in my local computer to do this task.
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