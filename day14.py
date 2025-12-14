import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import random

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("HR dataset.csv")

# Remove unwanted column
df.drop('Unnamed: 0', axis=1, inplace=True)

# Convert Hire_Date to datetime
df['Hire_Date'] = pd.to_datetime(df['Hire_Date'])

# -----------------------------
# Basic Data Exploration
# -----------------------------
print(df.info())
print(df['Performance_Rating'].unique())
print(df['Performance_Rating'].value_counts())
print(df['Performance_Rating'].mean())

print(df['Experience_Years'].unique())
print(df['Experience_Years'].value_counts())

# Experience distribution
sns.countplot(x='Experience_Years', data=df)
plt.show()

# -----------------------------
# Object & Numeric Columns
# -----------------------------
print(df.select_dtypes(include='object').head())
print(df.select_dtypes(include='number').head())

# -----------------------------
# Q1: Employee Status Distribution
# -----------------------------
status = df['Status'].value_counts()
status.plot(
    kind='pie',
    autopct='%1.1f%%',
    explode=(0.03, 0.03, 0.03, 0.03)
)
plt.title("Employee Status Distribution")
plt.show()

# -----------------------------
# Q2: Work Mode Distribution
# -----------------------------
work = df['Work_Mode'].value_counts()
work.plot(kind='pie', autopct='%1.1f%%', shadow=True)
plt.title("Work Mode Distribution")
plt.show()

# -----------------------------
# Q3: Employees per Department
# -----------------------------
sns.countplot(x='Department', data=df)
plt.show()

# -----------------------------
# Job Title Distribution
# -----------------------------
plt.figure(figsize=(10, 6))
sns.countplot(x='Job_Title', data=df)
plt.xticks(rotation=90)
plt.show()

# -----------------------------
# Q4: Average Salary by Department
# -----------------------------
dept_salary = df.groupby('Department')['Salary_INR'].mean() / 1000
dept_salary.plot(kind='bar', figsize=(10, 6))
plt.title("Average Salary by Department (in thousands)")
plt.ylabel("Salary")
plt.grid()
plt.show()

# -----------------------------
# Q5: Average Salary by Job Title
# -----------------------------
job_salary = df.groupby('Job_Title')['Salary_INR'].mean() / 1000
job_salary.plot(kind='bar', figsize=(10, 6))
plt.title("Average Salary by Job Title")
plt.xticks(rotation=90)
plt.grid()
plt.show()

# -----------------------------
# Q6: Avg Salary by Department & Job Title
# -----------------------------
dept_job_salary = df.groupby(
    ['Department', 'Job_Title']
)['Salary_INR'].mean() / 1000

colors = [f'#{random.randint(0, 0xFFFFFF):06x}' for _ in range(len(dept_job_salary))]
dept_job_salary.plot(kind='barh', figsize=(10, 8), color=colors)
plt.title("Average Salary by Department & Job Title")
plt.xlabel("Salary")
plt.show()

# -----------------------------
# Q7: Resigned & Terminated Employees
# -----------------------------
resigned = df[df['Status'] == 'Resigned']
terminated = df[df['Status'] == 'Terminated']

resigned_count = resigned.groupby('Department')['Status'].count()
terminated_count = terminated.groupby('Department')['Status'].count()

resigned_count.plot(kind='bar', color='black', label='Resigned')
terminated_count.plot(kind='bar', color='orange', label='Terminated')

plt.legend()
plt.ylabel("Employee Count")
plt.title("Resigned vs Terminated Employees")
plt.grid()
plt.show()

# -----------------------------
# Q8: Salary vs Experience
# -----------------------------
exp_salary = df.groupby('Experience_Years')['Salary_INR'].mean()
print(exp_salary)

# -----------------------------
# Q9: Avg Performance Rating by Department
# -----------------------------
perf = df.groupby('Department')['Performance_Rating'].mean()
plt.bar(perf.index, perf.values)
plt.title("Average Performance Rating by Department")
plt.ylabel("Rating")
plt.show()

# -----------------------------
# Q10: Country with Highest Employees
# -----------------------------
df['Country'] = df['Location'].apply(lambda x: str(x).split(',')[1])
print(df['Country'].value_counts().head(10))

# -----------------------------
# Q11: Correlation between Salary & Performance
# -----------------------------
print(df['Performance_Rating'].corr(df['Salary_INR']))

sns.heatmap(df.corr(numeric_only=True), annot=True)
plt.show()

# -----------------------------
# Q12: Hiring Trend Over Years
# -----------------------------
df['Year'] = df['Hire_Date'].dt.year
hire_trend = df.groupby('Year')['Employee_ID'].count()

hire_trend.plot(kind='bar', figsize=(10, 5))
plt.title("Hiring Trend Over Years")
plt.ylabel("Employees Hired")
plt.grid()
plt.show()

# -----------------------------
# Q13: Remote vs On-site Salary
# -----------------------------
print(df.groupby('Work_Mode')['Salary_INR'].mean())

# -----------------------------
# Q14: Top 10 Highest Paid Employees per Department
# -----------------------------
top_10 = df.groupby('Department').apply(
    lambda x: x.nlargest(10, 'Salary_INR'),
    include_groups=False
)
print(top_10)

# -----------------------------
# Q15: Attrition Rate by Department
# -----------------------------
dept_counts = df.groupby('Department')['Status'].agg(
    total_emp='count',
    resigned=lambda x: (x == 'Resigned').sum()
)

dept_counts['attrition_rate_%'] = (
    dept_counts['resigned'] / dept_counts['total_emp']
) * 100

print(dept_counts.sort_values('attrition_rate_%', ascending=False))
