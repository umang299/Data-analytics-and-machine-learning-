
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")


PATH = r"C:\Users\UMANG\BANKACAMP\bank-full.csv"
bank_df = pd.read_csv(PATH)

columns = bank_df.columns.tolist()[0].split(";")

rows = []
for i in range(0,len(bank_df)):
    rows.append(bank_df.iloc[i,:].values.tolist()[0].split(";"))

rows = np.array(rows)


bank_df = pd.DataFrame(rows,
            columns=columns)


x = bank_df['"y"'].value_counts().index.tolist()
y = bank_df['"y"'].value_counts().values.tolist()
y = [i/len(bank_df)*100 for i in y]


plt.figure(figsize = (8,5))
plt.ylim([0,100])
plt.bar(x,y,color = 'b',alpha = 0.5)
plt.grid()
plt.ylabel("Percentage $\%$")
plt.title("Percentage of people that subscribed a term deposit")
for x,y in zip([0,1],y):
    plt.text(x,y,y)


filt = bank_df['"y"'] == '"yes"'
subs = bank_df.loc[filt,:] # Dataframe of subscribers


sns.distplot(subs['age'])


x_job = subs['"job"'].value_counts().index.values
y_job = subs['"job"'].value_counts().values.tolist()
y_job = [i/len(subs)*100 for i in y_job]

x_marital = subs['"marital"'].value_counts().index.values
y_marital = subs['"marital"'].value_counts().values.tolist()
y_marital = [i/len(subs)*100 for i in y_marital]

x_edu = subs['"education"'].value_counts().index.values
y_edu = subs['"education"'].value_counts().values.tolist()
y_edu = [i/len(subs)*100 for i in y_edu]

x_def = subs['"default"'].value_counts().index.values
y_def = subs['"default"'].value_counts().values.tolist()
y_def = [i/len(subs)*100 for i in y_def]

x_loan = subs['"loan"'].value_counts().index.values
y_loan = subs['"loan"'].value_counts().values.tolist()
y_loan = [i/len(subs)*100 for i in y_loan]

x_house = subs['"housing"'].value_counts().index.values
y_house = subs['"housing"'].value_counts().values.tolist()
y_house = [i/len(subs)*100 for i in y_house]

plt.figure(figsize = (15,15))
plt.subplot(2,3,1)
plt.pie(y_job,labels = x_job )
plt.title("Employement Status")
plt.subplot(2,3,2)
plt.pie(y_marital , labels = x_marital)
plt.title("Marital Status")
plt.subplot(2,3,3)
plt.pie(y_edu , labels = x_edu)
plt.title("Educational Qualification")
plt.subplot(2,3,4)
plt.pie(y_def , labels = x_def)
plt.title("Credit Default Status")
plt.subplot(2,3,5)
plt.pie(y_loan , labels = x_loan)
plt.title("Personal Loan")
plt.subplot(2,3,6)
plt.pie(y_house , labels = x_house)
plt.title("Housing Loan")


x_cont = subs['"contact"'].value_counts().index.values
y_cont = subs['"contact"'].value_counts().values.tolist()
y_cont = [i/len(subs)*100 for i in y_cont]

x_month = subs['"month"'].value_counts().index.values
y_month = subs['"month"'].value_counts().values.tolist()
y_month = [i/len(subs)*100 for i in y_month]

x_po = subs['"poutcome"'].value_counts().index.values
y_po = subs['"poutcome"'].value_counts().values.tolist()
y_po = [i/len(subs)*100 for i in y_po]


plt.figure(figsize = (20,10))
plt.subplot(2,4,1)
plt.pie(y_cont , labels = x_cont)
plt.title("Contact Communication Type")
plt.subplot(2,4,2)
plt.pie(y_month , labels = x_month)
plt.title("Last contact month ")
plt.subplot(2,4,3)
plt.pie(y_po,labels = x_po)
plt.title("Last campaign outcome")
plt.subplot(2,4,4)
sns.distplot(subs['"day"'])
plt.title("Last contact day of month")
plt.subplot(2,4,5)
sns.distplot(subs['"duration"'])
plt.title("Last contact duration")
plt.subplot(2,4,6)
sns.distplot(subs['"campaign"'])
plt.title("Number of contacts during this campaign")
plt.subplot(2,4,7)
sns.distplot(subs['"pdays"'])
plt.title("Days from last contact of previous campaign")
plt.subplot(2,4,8)
sns.distplot(subs['"previous"'])
plt.title("Number of contacts before this campaign")



filt = bank_df['"y"'] == '"no"'
non_subs = bank_df.loc[filt,:] 


x_job = non_subs['"job"'].value_counts().index.values
y_job = non_subs['"job"'].value_counts().values.tolist()
y_job = [i/len(non_subs)*100 for i in y_job]

x_marital = non_subs['"marital"'].value_counts().index.values
y_marital = non_subs['"marital"'].value_counts().values.tolist()
y_marital = [i/len(non_subs)*100 for i in y_marital]

x_edu = non_subs['"education"'].value_counts().index.values
y_edu = non_subs['"education"'].value_counts().values.tolist()
y_edu = [i/len(non_subs)*100 for i in y_edu]

x_def = non_subs['"default"'].value_counts().index.values
y_def = non_subs['"default"'].value_counts().values.tolist()
y_def = [i/len(non_subs)*100 for i in y_def]

x_loan = non_subs['"loan"'].value_counts().index.values
y_loan = non_subs['"loan"'].value_counts().values.tolist()
y_loan = [i/len(non_subs)*100 for i in y_loan]

x_house = non_subs['"housing"'].value_counts().index.values
y_house = non_subs['"housing"'].value_counts().values.tolist()
y_house = [i/len(non_subs)*100 for i in y_house]


plt.figure(figsize = (15,15))
plt.subplot(2,3,1)
plt.pie(y_job,labels = x_job )
plt.title("Employement Status")
plt.subplot(2,3,2)
plt.pie(y_marital , labels = x_marital)
plt.title("Marital Status")
plt.subplot(2,3,3)
plt.pie(y_edu , labels = x_edu)
plt.title("Educational Qualification")
plt.subplot(2,3,4)
plt.pie(y_def , labels = x_def)
plt.title("Credit Default Status")
plt.subplot(2,3,5)
plt.pie(y_loan , labels = x_loan)
plt.title("Personal Loan")
plt.subplot(2,3,6)
plt.pie(y_house , labels = x_house)
plt.title("Housing Loan")


x_cont = non_subs['"contact"'].value_counts().index.values
y_cont = non_subs['"contact"'].value_counts().values.tolist()
y_cont = [i/len(non_subs)*100 for i in y_cont]

x_month = non_subs['"month"'].value_counts().index.values
y_month = non_subs['"month"'].value_counts().values.tolist()
y_month = [i/len(non_subs)*100 for i in y_month]

x_po = non_subs['"poutcome"'].value_counts().index.values
y_po = non_subs['"poutcome"'].value_counts().values.tolist()
y_po = [i/len(non_subs)*100 for i in y_po]


plt.figure(figsize = (20,10))
plt.subplot(2,4,1)
plt.pie(y_cont , labels = x_cont)
plt.title("Contact Communication Type")
plt.subplot(2,4,2)
plt.pie(y_month , labels = x_month)
plt.title("Last contact month ")
plt.subplot(2,4,3)
plt.pie(y_po,labels = x_po)
plt.title("Last campaign outcome")
plt.subplot(2,4,4)
sns.distplot(non_subs['"day"'])
plt.title("Last contact day of month")
plt.subplot(2,4,5)
sns.distplot(non_subs['"duration"'])
plt.title("Last contact duration")
plt.subplot(2,4,6)
sns.distplot(non_subs['"campaign"'])
plt.title("Number of contacts during this campaign")
plt.subplot(2,4,7)
sns.distplot(non_subs['"pdays"'])
plt.title("Days from last contact of previous campaign")
plt.subplot(2,4,8)
sns.distplot(non_subs['"previous"'])
plt.title("Number of contacts before this campaign")



jobs = bank_df['"job"'].value_counts().index.tolist()


def count(colname ,catname):
    filt = bank_df[str(colname)] == str(catname)
    number_y = bank_df.loc[filt,'"y"'].value_counts()[0]
    number_n = bank_df.loc[filt,'"y"'].value_counts()[1]
    return number_y , number_n


subs = []
non_subs = []
for j in jobs:
    num_y , num_n = count('"job"',j)
    total = num_y + num_n
    perc_subs = (num_y/total)*100
    subs.append(perc_subs)
    non_subs.append(100 - perc_subs)
   


plt.figure(figsize = (18,10))
plt.bar(jobs,subs,color = 'g',label = 'Subsriber')
plt.bar(jobs,non_subs,bottom=subs,color = 'r' , label = 'Non-Subscriber')
plt.xlabel("Employement Status",fontsize = 20)
plt.axhline(80,color = 'black',linewidth = 4 )
plt.ylabel("Percentage",fontsize = 20)
plt.ylim([0,100])
plt.legend()


marital= bank_df['"marital"'].value_counts().index.tolist()


subs = []
non_subs = []
for m in marital:
    num_y , num_n = count('"marital"',m)
    total = num_y + num_n
    perc_subs = (num_y/total)*100
    subs.append(perc_subs)
    non_subs.append(100 - perc_subs)

plt.figure(figsize = (18,10))
plt.bar(marital,subs,color = 'g',label = 'Subsriber')
plt.bar(marital,non_subs,bottom=subs,color = 'r' , label = 'Non-Subscriber')
plt.xlabel("Marital Status",fontsize = 20)
plt.axhline(80,color = 'black',linewidth = 4 )
plt.ylabel("Percentage",fontsize = 20)
plt.ylim([0,100])
plt.legend()

edu= bank_df['"education"'].value_counts().index.tolist()

subs = []
non_subs = []
for e in edu:
    num_y , num_n = count('"education"',e)
    total = num_y + num_n
    perc_subs = (num_y/total)*100
    subs.append(perc_subs)
    non_subs.append(100 - perc_subs)


plt.figure(figsize = (18,10))
plt.bar(edu,subs,color = 'g',label = 'Subsriber')
plt.bar(edu,non_subs,bottom=subs,color = 'r' , label = 'Non-Subscriber')
plt.xlabel("Level of education",fontsize = 20)
plt.axhline(80,color = 'black',linewidth = 4 )
plt.ylabel("Percentage",fontsize = 20)
plt.ylim([0,100])
plt.legend()

balance = [float(i) for i in bank_df['"balance"']]

plt.figure(figsize = (25,10))
plt.subplot(2,2,1)
sns.violinplot(x=bank_df['"education"'], y=balance)
plt.subplot(2,2,2)
sns.violinplot(x=bank_df['"job"'], y=balance)
plt.xticks(rotation = 45)
plt.subplot(2,2,3)
sns.violinplot(x=bank_df['"marital"'], y=balance)
plt.subplot(2,2,4)
sns.violinplot(x=bank_df['"y"'], y=balance)


plt.figure(figsize = (25,5))
plt.subplot(1,3,1)
age = [int(i) for i in bank_df['age']]
sns.scatterplot(age,balance,hue = bank_df['"y"'])
plt.xlabel("Age")
plt.ylabel("Average Annual Bank Balance")
plt.subplot(1,3,2)
sns.scatterplot(age,balance,hue = bank_df['"education"'])
plt.xlabel("Age")
plt.ylabel("Average Annual Bank Balance")
plt.subplot(1,3,3)
sns.scatterplot(age,balance,hue = bank_df['"job"'])
plt.xlabel("Age")
plt.ylabel("Average Annual Bank Balance")


def med_sal(colname,catname):
    a = bank_df.loc[bank_df[str(colname)] == str(catname),'"balance"'].values.tolist()
    a = [float(i) for i in a]
    return np.median(a)

med_Sal = []
for j in jobs:
    med_Sal.append(med_sal('"job"',j))


plt.figure(figsize = (20,10))
sns.barplot(jobs,med_Sal,palette= 'inferno')
plt.xlabel("Employement Status",fontsize = 18)
plt.ylabel("Median Salary",fontsize = 18)




