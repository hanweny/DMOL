# DMOL (Dynamic Motivation during Online Learing)
  This repo contains the work we are doing to understand the ways in which dynamics of motivation on a weekly and daily basis alter learning trajectories of online learners in a Biological Sciences course at UCI. All the identifiers are removed for the purpose of analysis. If you would like to discuss the work, and would like to be involved, please email me at shafeem@uci.edu.
  
# Our team:
 #### Lead Researcher - Shafee Mohammed (shafeem@uci.edu)
  
 #### Research Assistants - 
  Hanwen 'Henry' Ye
  
  Caleb Pitts
  
  Fanbo Shi
  
We have a set of 1460 variables in total that look at the students' prior performances, background/demographic details, click behavior on canvas learning platform, and a plethora of motivational constructs measured over the course of a few weeks in Summer 2018.


This work ties back in to the larger project, Investigating Virtual Learning Environments(IVLE) conducted at UC Irvine's School of Education. IVLE aims to investigate the learning processes of online learning, the associated teacher and learner's experiences, and the ways in which these translate to the students' learning outcomes.

Details of the larger project can be found [here](https://www.digitallearninglab.org/investigating-virtual-learning-environments/).
For more information contact Mark Warschauer at markw@uci.edu.

#### Description of each folder:

Clustering: uses LDA (Latent Dirichlet Allocation) to check possible activity-related response topics

Data:

Preprocessing:variable selections based on codebook review

Quiz prediction: predicting wether a quiz score will be above average or below average. Used nonlinear PCA to reduce the dimensionality and then used random forrest to classify students and preiduct the end results. 

Utils: 
