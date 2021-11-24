# NLP-Web-based-Data-Application

## Overview
In Milestone I, we have built several machine learning models based on different feature representations of documents to classify job advertisement categories. In Milestone II, we will develop a T
In this Project, we developed a job hunting website based on the Python Flask web development framework. The developed website will allow job hunters to browse existing job advertisements, as well as employers to create new job advertisements. The website will adopt one of the machine learning models built using the NLP pipeline, for the purpose of auto classifying categories for new job advertisement entries. Such functionality helps to reduce human data entry error, increase the job exposure to relevant candidates, and also improve the user experience of the job hunting site.

## Functionalities include:
### Job Display
In the homepage, the develop website should provide a list view of the available job advertisement previews, and when the users click on the previews, they will be able to see the full description of the job advertisement. 

### Job Search
The Job hunting website will allow job hunters to effectively search for job listings that are of their interest. The search could be based on keywords. Upon user entering a keyword string, the develop system should return a message saying how many matched job advertisements, and it also returns a list of job advertisement previews that are relevant to the keyword string. When users click on a job ad preview, they can see the full description of the job using the search algorithm that we designed:
* support to search keyword strings in similar forms. For example, if users enter the keyword strings “work” or “works” or “worked”, the search results from these two keyword strings will be the same.

### Create a New Job Listing
The Job hunting website will allow employers to create a new job listing, including entering various information, e.g., title, description, salary, etc.
Based on the entered title and description, the website should automatically recommend categories of the entered job advertisement. It should also allow the employer to select other categories if the recommendation does not suit (i.e. they can overwrite the category suggested by the website).
