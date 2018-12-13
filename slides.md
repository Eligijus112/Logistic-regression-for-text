Logistic regression in NLP using R
========================================================
author: Eligijus Bujokas
date: 2018.12.18
autosize: true



Content
========================================================

- What is natural language procesing
- NLP definitions
- Logistic regression in NLP tasks
- Quora example

Natural language procesing
========================================================

Natural language processing (NLP) is a subfield of computer science concerned with the interactions between computers and human (natural) languages, in particular how to program computers to process and analyze large amounts of natural language data.

- Classifying text
- Chatbots
- Word embeddings 

NLP definitions
========================================================

Corpus

A text corpus is a collection of texts of written (or spoken) language presented in electronic
form.

A document is the row of a matrix 

A term is single token from the corpus

NLP definitions
========================================================

Tokens

Eligijus is presenting now. 

[Eligijus], [is], [presenting], [now]

[Eligijus is], [presenting now]

NLP definitions
========================================================

N - grams

An n-gram is a contiguous sequence of n items from a given sequence of text. 

2 - gram (bigram):

[Eligijus is], [is presenting], [presenting now]

3 - gram: 

[Eligijus is presenting], [is presenting now]

Corpus in a matrix 
========================================================

Let us say we have a simple corpus:


| index|text                      |
|-----:|:-------------------------|
|     1|MIF is cool cool          |
|     2|opel ascona 2005m is cool |
|     3|what did the girl want    |

The document term matrix would then look like:


| did| want| ascona| 2005m| the| what| MIF| opel| girl| is| cool|
|---:|----:|------:|-----:|---:|----:|---:|----:|----:|--:|----:|
|   0|    0|      0|     0|   0|    0|   1|    0|    0|  1|    2|
|   0|    0|      1|     1|   0|    0|   0|    1|    0|  1|    1|
|   1|    1|      0|     0|   1|    1|   0|    0|    1|  0|    0|

DTM with 2 - grams
========================================================


| index|text                      |
|-----:|:-------------------------|
|     1|MIF is cool cool          |
|     2|opel ascona 2005m is cool |
|     3|what did the girl want    |


| cool_cool| the_girl| ascona_2005m| girl_want| 2005m_is| did_the| MIF_is| opel_ascona| what_did| is_cool|
|---------:|--------:|------------:|---------:|--------:|-------:|------:|-----------:|--------:|-------:|
|         1|        0|            0|         0|        0|       0|      1|           0|        0|       1|
|         0|        0|            1|         0|        1|       0|      0|           1|        0|       1|
|         0|        1|            0|         1|        0|       1|      0|           0|        1|       0|

Logistic regression in NLP context
========================================================

Let us say we have n documents and k unique tokens. Each document is assigned a 
binary label - {0, 1}. 

Then the general model is: 

$$ log \dfrac{P(Y = 1)}{P(Y = 0)} = \sum_{i = 0}^{k}\beta_{i} X_{i} \, (1)$$

We start indexing from zero because of the intercept. 

We get the coefficients $\widehat{\beta}$ by maximizing: 

$$  \sum_{i = 1}^{n}\left(y_{i}(\beta_{0} + \beta^T x_{i}) - log(1 + e^{\beta_{0} + \beta^T x_{i}})) \right)  $$

in respect to $\beta_{0}$ and $\beta$


Lasso and Ridge logistic regression
========================================================

Ridge (more uniform distribution of coefficient values): 

$$ \dfrac{1}{N} \sum_{i = 1}^{n}\left(y_{i}(\beta_{0} + \beta^T x_{i}) - log(1 + e^{\beta_{0} + \beta^T x_{i}})) \right) - \alpha \sum_{j = 1}^{k} \beta_{j} ^ 2$$

Lasso (reduces feature number): 

$$ \dfrac{1}{N} \sum_{i = 1}^{n}\left(y_{i}(\beta_{0} + \beta^T x_{i}) - log(1 + e^{\beta_{0} + \beta^T x_{i}})) \right) - \alpha \sum_{j = 1}^{k} |\beta_{j}| $$

Elastic net (something in between): 

$$ \dfrac{1}{N} \sum_{i = 1}^{n}\left(y_{i}(\beta_{0} + \beta^T x_{i}) - log(1 + e^{\beta_{0} + \beta^T x_{i}})) \right) - (\alpha \sum_{j = 1}^{k} |\beta_{j}| + (1 - \alpha) \sum_{j = 1}^{k} \beta_{j} ^ 2)$$

Quora insincere question competition
========================================================

Competition: 

https://www.kaggle.com/c/quora-insincere-questions-classification 

Data: 

https://www.kaggle.com/c/quora-insincere-questions-classification/data

Y = 1 if the question is not sincere

Y = 0 if the question is sincere

Example
========================================================


|question_text                                                                     | target|
|:---------------------------------------------------------------------------------|------:|
|How did Quebec nationalists see their province as a nation in the 1960s?          |      0|
|Do you have an adopted dog, how would you encourage people to adopt and not shop? |      0|
|Why does velocity affect time? Does velocity affect space geometry?               |      0|
|How did Otto von Guericke used the Magdeburg hemispheres?                         |      0|


|question_text                                                                                                                             | target|
|:-----------------------------------------------------------------------------------------------------------------------------------------|------:|
|Has the United States become the largest dictatorship in the world?                                                                       |      1|
|Which babies are more sweeter to their parents? Dark skin babies or light skin babies?                                                    |      1|
|If blacks support school choice and mandatory sentencing for criminals why don't they vote Republican?                                    |      1|
|I am gay boy and I love my cousin (boy). He is sexy, but I dont know what to do. He is hot, and I want to see his di**. What should I do? |      1|



Ridge regression results
========================================================


```r
dim(dtm)
```

```
[1] 161620  10957
```




```r
dim(coef_table)
```

```
[1] 10958     2
```

Ridge regression results (sincere sentiment)
========================================================


|features      | coefficient|
|:-------------|-----------:|
|nc            |   -3.043594|
|loneliness    |   -3.069235|
|habitat       |   -3.104099|
|pains         |   -3.131535|
|immortality   |   -3.202438|
|aftermath     |   -3.203054|
|responded     |   -3.241714|
|ipcc          |   -3.491197|
|leap          |   -3.658487|
|trades        |   -3.759274|
|curved        |   -3.951417|
|independently |   -3.969513|
|whey          |   -4.575782|

Ridge regression results (insincere sentiment)
========================================================


|features    | coefficient|
|:-----------|-----------:|
|mindless    |    5.110071|
|castration  |    4.871917|
|alabamians  |    4.805299|
|castrating  |    4.699088|
|tennesseans |    4.629572|
|brandeis    |    4.505227|
|castrated   |    4.326945|
|nigerians   |    4.290410|
|satanism    |    4.246887|
|isaiah      |    4.149295|
|nerds       |    4.112068|
|fanboys     |    4.100346|
|castrate    |    4.071429|

Lasso regression results
========================================================




```r
dim(coef_table)
```

```
[1] 267   2
```

Lasso regression results (sincere sentiment)
========================================================


|features    | coefficient|
|:-----------|-----------:|
|book        |  -0.1769008|
|exam        |  -0.1862232|
|career      |  -0.1872319|
|tips        |  -0.1950791|
|job         |  -0.2074193|
|computer    |  -0.2272395|
|app         |  -0.2315998|
|online      |  -0.2644002|
|company     |  -0.3070122|
|study       |  -0.3219155|
|difference  |  -0.3856541|
|engineering |  -0.5246687|
|(Intercept) |  -0.7998757|

Lasso regression results (insincere sentiment)
========================================================


|features  | coefficient|
|:---------|-----------:|
|liberals  |    1.930906|
|indians   |    1.856574|
|trump     |    1.817980|
|muslims   |    1.795541|
|americans |    1.763194|
|castrated |    1.608803|
|democrats |    1.591508|
|women     |    1.552220|
|girls     |    1.520034|
|jews      |    1.385587|
|gay       |    1.359668|
|atheists  |    1.343609|
|castrate  |    1.324343|

