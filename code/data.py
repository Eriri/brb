import pandas as pd

df = pd.read_csv("../data/titanic/test.csv", header=0)
df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Fare', 'Cabin'])

f = open("../data/titanic/test_rev.txt", "w")

for i, d in df.iterrows():
    f.write(str(d["Pclass"])+" "+str(d["Sex"])+" " +
            str(d["Age"])+" "+str(d["SibSp"])+" "+str(d["Parch"])+" "+str(d["Embarked"])+"\n")
