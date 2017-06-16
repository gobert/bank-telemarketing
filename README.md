This is based on the [dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing) that is used for the paper " [A data-driven approach to predict the success of bank telemarketing](https://scholar.google.fr/scholar?q=A+data-driven+approach+to+predict+the+success+of+bank+telemarketing) ".

The idea is to build a classifier that will predict who is going to purchase the product.

IMHO it's a good dataset for training oneself in M.L. by building one classifier. After you can compare the results with the paper.


## Notebooks
* data-analyze.ipynb: Let's start with some data analysis of the dataset!

## Dataset
### Content
Input variables:

**bank client data:**

 1. age (numeric) job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')
 2. marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)
 3. education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')
 4. default: has credit in default? (categorical: 'no','yes','unknown')
 5. housing: has housing loan? (categorical: 'no','yes','unknown') loan:
 6. has personal loan? (categorical: 'no','yes','unknown')

**related with the last contact of the current campaign:**

 7.  contact: contact communication type (categorical: 'cellular','telephone')
 8. month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec')
 9. day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')
 10. duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.

**other attributes:**

 11. campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
 12. pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
 13. previous: number of contacts performed before this campaign and for this client (numeric)
 14. poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')


**social and economic context attributes**

 15. emp.var.rate: employment variation rate - quarterly indicator (numeric)

 16. cons.price.idx: consumer price index - monthly indicator (numeric)

 17. cons.conf.idx: consumer confidence index - monthly indicator (numeric)

 18. euribor3m: euribor 3 month rate - daily indicator (numeric)

 19. nr.employed: number of employees - quarterly indicator (numeric)

**Output variable (desired target):**

 20. y - has the client subscribed a term deposit? (binary: 'yes','no')


## Citation Request

This dataset is public available for research. The details are described in [Moro et al., 2014].
Please include this citation if you plan to use this database:

[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014
