# SUMOTwitterInsights
Cultures of Experimentation workshop 2018 Round 5 initiative to analyze open text of Mozilla Support @firefox customer support sentiment 

# What is this project about

This project is designed to analyze open text customer support sentiment. 

# Hypothesis and Background
Mozilla Support creates release reports during the first three weeks of a Firefox Desktop release. It included sentiment and summaries of user issue from one to one support channels that are prioritized by the Mozilla Support team (SUMO). These open text data sets include: Mozilla forum questions and answers and Twitter conversations, and CSAT open text survey sentiment. Right now, this data is rich in information from users that are experiencing the headaches that could be improved upon in the product. This experiment is to enhance the presentation of these issues and summarize them in a more meaningful way with text analysis. 

Working with Ben, we would like to analyze the open text to get an idea of positive and negative words based on NLTK library to see if there are any new insights that the current model does not show. 

Hypothesis: Based on two sample csv data dumps from each open text data in each one to one support channel we want to see if the sentiment library reveals any new issues with the new model. By the end of the 2h we will have a sample report of positive and negative user issue trends based on the english library and be able to tell you if the model shows any other pain points from users looking for help with Firefox. (that were not revealed by previous tagging methods https://docs.google.com/spreadsheets/d/105UYPWdTncoddmdL8rrXGM8uSDUG7XSVLVKzTSJmMMM/edit#gid=0 in the Reply by Buffer Tool Mozilla uses to answer @firefox support tweets)


## Desired use case

# Later - webpage link, presentation
# People
Data Analyst: Ben
Collaborators: Rachel, Madalina, Roland

# Data description
The first version of this project will include the tags that have been manually added and programed in the Reply by Buffer tool and the open text tweet from the support conversations inbox. These are considered the 'support conversations' sentiment that we would like to analyze for user issues further. 

# Version 1
Version 1: 
Trains the model to analyze tweets and classifies words from a sample set from Kaggle to identify positive and negative works in twitter. 
The model then applies what it learned to the csv sample set from the Reply by Buffer tool export. Tags and Tweet text are only included in this csv file sample. 
Th emodel then prints out words with the most frequency and the ratio between the postive and negative tweet frequency to give more detail about the sentiment of the support tweets. 

# Version 2 Use Cases
As a reporter for the Support Community Release report, I want to be able to see positive and negative summaries for the different languages that are offered community support in twitter for Mozilla's Social Support Team. 
As a interpreter of this report, I want to be able to ask, what tweets are contributing the negativity to this keyword - or tag that describes the user issue?, what are the biggest pain points of the tweets that increased from one week to the next, and what are the tweets displayed for each postive or negative word? 
(More being added week of Oct 8 - Oct 20)

Version 2 model - Jupyter Notebook Display desired. 


# Requirements and Library Versions
*Python version (3)
*Ntlk library 
(included = what is the terminal link that installs these)

<later>
<sub>Rough project timeline and links: <>
Oct
Use case: Feed a tweet: is it negative or positive
* Input: csv - yellow columns text, tag, size = 1k (weeks worth )
* Output = add column that says positive or negative
</sub>
Nov
2nd iteration add the language 
Use cases based 1st iteration = “next steps” slide

End of Nov - have prototype website that summarizes week work of tweets
https://metrics.mozilla.com/protected/bmiroglio/firefox-addons/_site/topline.html#what-percent-of-users-are-on-57 
  
# How to use: 

Command to run script

# What it should look like - results 

# Conclusion Post mortem 

</later>
