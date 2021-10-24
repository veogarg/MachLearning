# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 14:23:09 2021

@author: Nishan Kapoor
"""
######### Question 1 ########

# Implementing Apriori algorithm from mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


df = pd.read_csv(r"E:\Association rules Dataset\book.csv")

frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 5)), height = frequent_itemsets.support[0:5], color ='rgmyb')
plt.xticks(list(range(0, 5)), frequent_itemsets.itemsets[0:5], rotation=30)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part #################################

def to_list(i):
    return (sorted(list(i)))

ma_df = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_df = ma_df.apply(sorted)

rules_sets = list(ma_df)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

#############################################################################
#############################################################################
######### Question 2 ########

# Implementing Apriori algorithm from mlxtend


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

groceries = []
with open(r"E:\Association rules Dataset\groceries.csv") as f:
    groceries = f.read()

# splitting the data into separate transactions using separator as "\n"
groceries = groceries.split("\n")

groceries_list = []
for i in groceries:
    groceries_list.append(i.split(","))

all_groceries_list = [i for item in groceries_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_groceries_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[5:15], x = list(range(5, 15)), color = 'rgbkymc')
plt.xticks(list(range(5, 15), ), items[5:15])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
groceries_series = pd.DataFrame(pd.Series(groceries_list))
groceries_series = groceries_series.iloc[:9835, :] # removing the last empty transaction

groceries_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = groceries_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 7)), height = frequent_itemsets.support[0:7], color ='rgmyk')
plt.xticks(list(range(0, 7)), frequent_itemsets.itemsets[0:7], rotation=25)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)
#############################################################################
#############################################################################

######## Question 3 #############

# Implementing Apriori algorithm from mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


df = pd.read_csv(r"E:\Association rules Dataset\my_movies new.csv")

frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 5)), height = frequent_itemsets.support[0:5], color ='rgmyb')
plt.xticks(list(range(0, 5)), frequent_itemsets.itemsets[0:5], rotation=30)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part #################################

def to_list(i):
    return (sorted(list(i)))

ma_df = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_df = ma_df.apply(sorted)

rules_sets = list(ma_df)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

#############################################################################
#############################################################################

######## Question 4 #############

# Implementing Apriori algorithm from mlxtend

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt


df = pd.read_csv(r"E:\Association rules Dataset\myphonedata new.csv")

frequent_itemsets = apriori(df, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 5)), height = frequent_itemsets.support[0:5], color ='rgmyb')
plt.xticks(list(range(0, 5)), frequent_itemsets.itemsets[0:5], rotation=30)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(10)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part #################################

def to_list(i):
    return (sorted(list(i)))

ma_df = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_df = ma_df.apply(sorted)

rules_sets = list(ma_df)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)

#############################################################################
#############################################################################

########## Question 5 ############

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

retail = []
with open(r"E:\Association rules Dataset\transactions_retail1.csv") as f:
    retail = f.read()


# splitting the data into separate transactions using separator as "\n"
retail = retail.split("\n")

retail_list = []
for i in retail:
    retail_list.append(i.split(","))

all_retail_list = [i for item in retail_list for i in item]

from collections import Counter # ,OrderedDict

item_frequencies = Counter(all_retail_list)

# after sorting
item_frequencies = sorted(item_frequencies.items(), key = lambda x:x[1])

# Storing frequencies and items in separate variables 
frequencies = list(reversed([i[1] for i in item_frequencies]))
items = list(reversed([i[0] for i in item_frequencies]))

# barplot of top 10 
import matplotlib.pyplot as plt

plt.bar(height = frequencies[3:9], x = list(range(3, 9)), color = 'rgbkymc')
plt.xticks(list(range(3, 9), ), items[3:9])
plt.xlabel("items")
plt.ylabel("Count")
plt.show()


# Creating Data Frame for the transactions data
retail_series = pd.DataFrame(pd.Series(retail_list))
retail_series = retail_series.iloc[:557042, :] # removing the last empty transaction

retail_series.columns = ["transactions"]

# creating a dummy columns for the each item in each transactions ... Using column names as item name
X = retail_series['transactions'].str.join(sep = '*').str.get_dummies(sep = '*')

frequent_itemsets = apriori(X, min_support = 0.0075, max_len = 4, use_colnames = True)

# Most Frequent item sets based on support 
frequent_itemsets.sort_values('support', ascending = False, inplace = True)

plt.bar(x = list(range(0, 11)), height = frequent_itemsets.support[0:11], color ='rgmyk')
plt.xticks(list(range(0, 11)), frequent_itemsets.itemsets[0:11], rotation=20)
plt.xlabel('item-sets')
plt.ylabel('support')
plt.show()

rules = association_rules(frequent_itemsets, metric = "lift", min_threshold = 1)
rules.head(20)
rules.sort_values('lift', ascending = False).head(10)

################################# Extra part ###################################
def to_list(i):
    return (sorted(list(i)))

ma_X = rules.antecedents.apply(to_list) + rules.consequents.apply(to_list)

ma_X = ma_X.apply(sorted)

rules_sets = list(ma_X)

unique_rules_sets = [list(m) for m in set(tuple(i) for i in rules_sets)]

index_rules = []

for i in unique_rules_sets:
    index_rules.append(rules_sets.index(i))

# getting rules without any redudancy 
rules_no_redudancy = rules.iloc[index_rules, :]

# Sorting them with respect to list and getting top 10 rules 
rules_no_redudancy.sort_values('lift', ascending = False).head(10)






