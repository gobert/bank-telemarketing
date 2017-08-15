# sem-marketing

Search Engine Marketing (SEM) is a form of Internet marketing that involves the promotion of websites by increasing their visibility in search engine results through paid advertising. For instance: paying google to appear in the results when the user search "*buy book 'Data Science Essentials in Python'*".

The aim of this work is about predicting the outcome of a marketing campaign: for each bought keyword, the system will predict the customer's expense on the website. If the prediction is accurate, then one operator can set the price for each campaign accordingly.


## Dataset
You need to [download the file](https://drive.google.com/file/d/0B1BAeQsIi6lgc0RiQlc1RHpiTFE/view?usp=sharing) and then to extract it in the folder of this file: `tar xzvf sem-database.cs.tgz`.

The dataset is based on real data, but it has been ciphered for privacy reason:
* There is hashed data instead of labeled data;
* dates of the data have been moved (but they kept in the same order);
* prices have been converted to an imaginary currency.

### Content

**Input variables:**
* Date - Date when the campaign happened
* [Keyword_ID](https://support.google.com/adwords/answer/6323?hl=en)
* [Ad_group_ID](https://support.google.com/adwords/answer/6298?hl=en) - In this dataset all keywords of the same product have the same ad group ; and all keywords in an ad group are about the same product.
* [Campaign_ID](https://support.google.com/adwords/answer/6304?hl=en)
* [Account_ID](https://support.google.com/adwords/answer/17779?hl=en)
* Device_ID - eg: computers, smartphones, tablets, ...
* [Match_type_ID](https://support.google.com/adwords/answer/2497836?hl=en)


**Output variables**

* Clicks - How many times the campaign have been clicked = How many customers arrived on the website
* Conversions - Conversion rate of the campaign = which % of the customers have bought a product
* Revenue - How much have the campaign earn

**Prediction variable (desired output)**

* Revenue per Clicks (RPC) = Revenue / Clicks


## Concepts of the predictive model
Online marketing is about solving 2 main problems:

1. Solving a traffic problem: Is our product known? Do people arrive on our website? This can be solved in spending more money on SEM.
2. Conversion problem: how many of the customers that arrive on our website will buy the product. This can be solved, for instance, in having a better design for the website.

![High level schema of the system](https://user-images.githubusercontent.com/1684807/28922361-26f4fb14-785a-11e7-8e56-d94fe8360d3e.png)

The model is divided in 2 sub-modules:
1. The acquisition model: how many visits are going to be acquired today?
2. The conversion model: given n customers on our website, how many are going to buy a product (aka how many are going to be converted to clients).
