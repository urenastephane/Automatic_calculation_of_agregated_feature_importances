# Automatic_calculation_of_agregated_feature_importances
Automated script to get readable feature importances (for categorical values transformed with OHE).

What's the matter?
This script was designed for:
-Getting an automated process to directly create pipelines from a list of features provided 
(In this example, there is a little hack since certain values are not strings (int or double) and are treated as indexed values.
As my df columns had those types I did not change them. But it could be easier to treat numerical values as double types for instance 
and strings, on the other hand, to index them in a same loop (simpler))

-Then I create an algorithm to agregate feature importances 
(which are difficult to read when you have lots of categorical values with thousands of categories...
Not to mention the time it takes)

I hope this will help people not to lose time with pipelines and easily read feature importances (when dealing with lots of categorical features)
