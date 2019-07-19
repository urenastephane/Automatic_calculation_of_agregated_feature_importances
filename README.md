# Automatic_calculation_of_agregated_feature_importances
Automated script to get readable feature importances (for categorical values transformed with OHE).

What's the matter?
This script was designed for:  
1) Getting an automated process to directly create pipelines from a list of features provided 
(In this example, there is a little hack since some values are not strings (int or double) but are categorical values. To make the task easier (and the algorithm much simplier), choose to encode numerical features with double type and categorical with StringType(). 


2) Then I create an algorithm to agregate feature importances 
(which are difficult to read when you have lots of categorical values with thousands of categories...
Not to mention the time it takes)

I hope this will help people not to lose time with pipelines and easily read feature importances (when dealing with lots of categorical features)
