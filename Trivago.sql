
--- Task 1---- To return Ten most searched destinations for each 10 minute interval starting 00:00:00------
select distinct(A.hotel), strftime('%M:%S', CAST ((julianday(A.time) - julianday(B.time)) AS REAL), '12:00') as time_diff
from Trivago A, Trivago B
where A.user_id <> B.user_id and
time_diff <= '10:00' LIMIT 10;

---Task 2----To get distinct hotel clicked on by the user--------------
select distinct hotel, user_id
from Trivago
group by hotel,user_id
having count(user_id) >1

---Task 3-----Inserting the result of Task 1 and Task 2 in a table-------------
Insert into TrivDist (Hotel, User_Id)
select distinct hotel, user_id
from Trivago
group by hotel,user_id
having count(user_id) >1;

Insert into TrivTime (Hotel,Tdiff)
select distinct(A.hotel), strftime('%M:%S', CAST ((julianday(A.time) - julianday(B.time)) AS REAL), '12:00') as time_diff
from Trivago A, Trivago B
where A.user_id <> B.user_id and
time_diff <= '10:00' and A.hotel is not NUll
group by A.hotel
LIMIT 10;