

Netflix Movie Recommendation System

RAGUL VENKATARAMAN RAVISANKAR, University of Colorado Boulder, USA

ARAVINDAKUMAR VIJAYASRI MOHAN KUMAR, University of Colorado Boulder, USA

RAWSHON FERDAWS, University of Colorado Boulder, USA

1

INTRODUCTION

Recommender Systems are a way of predicting what rating or preference a user would give to an

item. During the last few decades, with the rise of YouTube, Amazon, Netﬂix and many other such

web services, recommender systems have become more vital. Recommender systems can generate

huge revenue when they are eﬃcient. It is also a way to stand out signiﬁcantly from competitors. In

this project, we will take the Netﬂix prize dataset and build a recommender system to recommend

movies to the users based on various techniques.

2

RELATED WORK

There are various techniques to recommend movies to the users.

2.1 Collaborative-Filtering

This technique ﬁlters information by using the interactions and data collected by the system from

other users. It is based on the idea that people who agreed in their evaluation of certain items

are likely to agree again in the future. Collaborative-ﬁltering systems focus on the relationship

between users and items. There are two classes of Collaborative-Filtering:

(1) User-User Similarity This measures the similarity between target users and other users.

With the similarity score we can compare each user among the rest of n - 1 users. The higher

the similarity between vectors, the higher the similarity between users. The similarity matrix

(U) for n users is represented by







1

.

.

.

1 2

1

.

1

.

1



푠푖푚( , )... 푠푖푚( ,푖)... 푠푖푚( ,푛)







2



푠푖푚( ,푛)









.

.

.

.

1





푈 =



1

.



.









.

.







0

1



푠푖푚(푛, ) 푠푖푚(푛, ).. 푠푖푚(푛, 푖)..

(2) Item-Item Similarity This measures the similarity between the items that target users rate

or interact with and other items. The similarity matrix (I) from m items is represented by

Authors’ addresses: Ragul Venkataraman Ravisankar, University of Colorado Boulder, Boulder, USA, rave2101@colorado.edu;

Aravindakumar Vijayasri Mohan Kumar, University of Colorado Boulder, Boulder, USA, arvi7401@colorado.edu; Rawshon

Ferdaws, University of Colorado Boulder, Boulder, USA, rafe7973@colorado.edu.





2

Ragul Venkataraman Ravisankar, Aravindakumar Vijayasri Mohan Kumar, and Rawshon Ferdaws







1

.

.

.

1 2

1

.

1

.

1



푠푖푚( , )... 푠푖푚( ,푖)... 푠푖푚( ,푚)







2



푠푖푚( ,푚)









.

.

.

.

1





퐼 =



1

.



.









.

.







0

1



푠푖푚(푚, ) 푠푖푚(푚, ).. 푠푖푚(푚, 푖)..

3

PROPOSED WORK

We are planning to take the outputs from the multiple baseline models (built on user-user similarity,

SVD etc) and feed them as inputs into our recommender model and analyze how the accuracy can

be improved. We are planning to use the [Netﬂix](https://www.kaggle.com/netflix-inc/netflix-prize-data)[ ](https://www.kaggle.com/netflix-inc/netflix-prize-data)[Prize](https://www.kaggle.com/netflix-inc/netflix-prize-data)[ ](https://www.kaggle.com/netflix-inc/netflix-prize-data)[dataset](https://www.kaggle.com/netflix-inc/netflix-prize-data)

Fig. 1. Proposed System

4

CHALLENGES

4.1 Cold Start Problem

Recommender Systems generally recommends items to the user based on the historical data. When

a new user signs up to the product there won’t be any historical data about that user. This is called

the Cold-Start problem. The cold-start problem is usually handled using Content-based ﬁltering

techniques. For example, recommending the most popular items to the new users.

4.2 Business Metric

In Recommender Systems, there are no standard techniques to measure how good the system

performs. Deﬁning a proper business metric in such scenarios to evaluate the accuracy of the

system is challenging.





Netflix Movie Recommendation System

3

5

DATA PREPROCESSING

• The given user-movie rating dataset was in a text ﬁle format we converted it into the csv ﬁle

to read it as a dataframe (User, Movie, Rating, Rating Date).

• Data Cleaning was not required as there were no empty values.

• We split them into train test data based on an 80:20 split ratio.

• We tried to represent the given data in the form of a sparse matrix, because there are a lot of

zero values as users would have watched only a handful of movies. Therefore, for most of

the movies the ratings won’t be present. To utilize the space eﬃciently, we went with the

sparse matrix representation.

• We then compute user-user similarity (how the given user is similar to all the other users)

and movie-movie similarity (how the given movie is similar to all the other movies) using

cosine similarity.

6

EXPLORATORY DATA ANALYSIS

For the given training dataset we took basic statistics of the number of unique users, movies and

ratings. There were around 405k users, 17k Movies and 100M ratings. The histogram of ratings for

the training data set is shown below:

Fig. 2. Distribution of ratings over Training dataset

We also tried to analyze if the day of the week has any signiﬁcance on the ratings. But on plotting

the graph, we found that the above feature had no signiﬁcance.





4

Ragul Venkataraman Ravisankar, Aravindakumar Vijayasri Mohan Kumar, and Rawshon Ferdaws

Fig. 3. Ratings given on each day of the week

Fig. 4. Box Plot of Days of Week

The probability density function (PDF) was obtained for user-average and movie-average ratings.

From the graph we can see that, they are not following a normal distribution but a skewed

distribution.





Netflix Movie Recommendation System

5

Fig. 5. User Average and Movie Average Ratings

7

FEATURE ENGINEERING

From the given user-movie matrix we are trying to derive features and pose it as a regression

problem. There are thirteen handcrafted features.

• Global Average Rating

• User Average Rating

• Movie Average Rating

• Top 5 similar user rating

• Top 5 similar movie rating

We will feed the above features as input to the XgBoost baseline model.

8

ARCHITECTURE

We are using the 13 handcrafted features for the XgBoost baseline model and other Surprise Baseline

models, and then take the outputs of each model and give as features to the ﬁnal XgBoost model

along with the 13 handcrafted features.

8.1 XgBoost Baseline

XgBoost is an open-source library, which implements high-performance Gradient-Boosted Decision

Trees(GBDT). The underlying implementation is using C++ and on top of that there is a Python

wrapper making it very fast for regression problems.

We feed the 13 handcrafted features to the XgBoost baseline model.





6

Ragul Venkataraman Ravisankar, Aravindakumar Vijayasri Mohan Kumar, and Rawshon Ferdaws

8.2 Surprise Baseline

Surprise is a Python scikit library for building and analyzing recommender systems that deal with

explicit rating data.

In the Surprise Baseline model we will try to predict the movie rating based on global average,

user’s average rating and movie’s average rating received.

푟ˆ푢푖 = 푏푢푖 = 휇 + 푏푢 + 푏푖

Equation Source: [https://surprise.readthedocs.io/en/stable/basic_algorithms.html](https://surprise.readthedocs.io/en/stable/basic_algorithms.html )

휇 : Global Average Rating

푏 : User bias

푢

푏 : Item bias

푖

푟ˆ : Predicted rating for a movie i by a user u

푢푖

8.3 Surprise KNN model

KNNBaseline is a collaborative ﬁltering algorithm which takes the baseline rating into consideration.

We built two models, one for users and one for movies.

• KNNBaseLineUser: We predict the rating based on the top-k similar users’ rating to that

movie (k is the hyper parameter here)

• KNNBaseLineMovie: We predict the ratings based on top-k similar movies to the given movie.

The prediction 푟ˆ푢푖 is set as:

~~Í~~

푠푖푚(푢,푣).(푟푣푖 − 푏푣푖)

푣∈푁

[~~Í](https://surprise.readthedocs.io/en/stable/knn_inspired.html)~~(~~

푘

푖

~~)~~( )

푢

푟ˆ푢푖 = 푏푢푖

\+

푠푖푚(푢,푣)

(푢)

(

푘

푖

)

푣∈푁

Equation Source: <https://surprise.readthedocs.io/en/stable/knn_inspired.html>

푟ˆ : Predicted rating for a movie i by a user u

푢푖

sim(u,v): Pearson similarity for u and v (u and v are either users or items)

8.4 Matrix Factorization

Matrix factorization is a collaborative ﬁltering algorithm which works by decomposing the user-item

interaction matrix into the product of two lower dimensionality rectangular matrices.

(1) Singular Value Decomposition (SVD)

퐴 = 푈푆푉푇

where

U, V: Orthogonal matrices with orthonormal eigenvectors

S :Diagonal matrix with square root of positive eigenvalues

The prediction 푟ˆ푢푖 is set as:

푟ˆ푢푖 = 휇 + 푏 + 푏 + 푞 푝

푇

푢

푖

푖

푢

Equation Source: <https://surprise.readthedocs.io/en/stable/matrix_factorization.html>





Netflix Movie Recommendation System

7

(2) SVD++ (or) SVDpp

It is an extension of SVD technique which also takes implicit ratings into consideration.

The prediction 푟ˆ푢푖 is set as:

[Õ](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)

[ꢀ](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)

[ꢁ](https://surprise.readthedocs.io/en/stable/matrix_factorization.html)

1

푟ˆ푢푖 = 휇 + 푏 + 푏 + 푞 푝푢 퐼푢 |−

푇

\+ |

푦푗

2

푢

푖

푖

푗 ∈퐼

푢

Equation Source: <https://surprise.readthedocs.io/en/stable/matrix_factorization.html>

Using techniques like SVD and SVD++ we will try to factor the input sparse matrix and

predict the ratings for a movie for a given user.

8.5 XgBoost Final Model

We take output from all the above models and in addition the 13 handcrafted features we provide

in total of 19 features to the XgBoost ﬁnal model and predict the rating.

9

EVALUATION

v

t

Õ

1

푛

2

푅푀푆퐸 =

(푆 − 푂 )

푖

푖

푛

푖=1

Si : Predicted values

Oi : Observed values

n : Number of samples

The performance of our model will be evaluated based on how close the predicted score is to

the actual score. The RMSE value should be as minimal as possible.

10 HYPERPARAMETER TUNING

Hyperparameter is a parameter on which the model performs it’s best for the given data. The

process of determining this ideal hyperparameter is called hyperparameter-tuning.

There are two types of hyperparameter tuning.

• Grid Search : The search operation is performed in a manually speciﬁed subset of hyperpa-

rameter space

• Random Search : Random combinations of hyperparameters are used to ﬁnd the best argument

in the given range

We performed the hyperparameter tuning using Grid Search technique on four models namely

KNN User, KNN Movie, SVD and SVDpp





8

Ragul Venkataraman Ravisankar, Aravindakumar Vijayasri Mohan Kumar, and Rawshon Ferdaws

11 RESULTS

11.1 XgBoost Baseline model Feature importance

From the plot below, we can observe that movieAvg has the highest feature importance for the

XgBoost baseline model followed by the userAvg. Similarity rating of the ﬁfth user (sur5) has the

least feature importance.

Fig. 6. XgBoost Baseline model Feature importance

11.2 XgBoost Final model Feature importance

From the plot below, we can observe that baseline\_xgb has the highest feature importance for the

XgBoost ﬁnal model followed by the movieAvg. Similarity rating of the second user (sur2) has the

least feature importance.

Fig. 7. XgBoost Final model Feature importance





Netflix Movie Recommendation System

9

11.3 RMSE Plot on Test Data

From the RMSE plot for the test data for all the diﬀerent models we observe that baseline\_xgb has

the least RMSE value followed by xgb\_ﬁnal. For the other models the RMSE values are slightly

higher than these two models.

Fig. 8. RMSE Metric on Test Data

11.4 MAPE plot on Test Data

Similar to the RMSE plot on test data the MAPE values for both baseline\_xgb and xgb\_ﬁnal are

the lowest. For the other models MAPE values are slightly higher than these two models.

Fig. 9. MAPE Metric on Test Data





10

Ragul Venkataraman Ravisankar, Aravindakumar Vijayasri Mohan Kumar, and Rawshon Ferdaws

11.5 RMSE Values for diﬀerent models

Fig. 10. RMSE Values for diﬀerent models

ACKNOWLEDGMENTS

We sincerely thank our Professor Qin Lv and teaching assistant Yichen Wang for this wonderful

opportunity. We had a great time exploring various techniques of recommender systems and we

were able to come up with an ensemble approach to give better recommendations. By submitting

this work we adhere to the honor code pledge: “On our honor, as University of Colorado

Boulder students, we have neither given nor received unauthorized assistance”

12 CODE CONTRIBUTIONS

• Aravindakumar Vijayasri Mohan Kumar: EDA, KNN user, KNN Movie, SVD++, Final

XgBoost, Hyperparameter Tuning, Documentation

• Ragul Venkataraman Ravisankar: Feature Engineering, XgBoost Baseline, Surprise Base-

line, Final XgBoost, Hyperparameter Tuning, Documentation

• Rawshon Ferdaws: Data preprocessing, EDA, SVD, Documentation

Git Hub Repository: [Netﬂix](https://github.com/vmaravind96/NetflixMovieRecommendation)[ ](https://github.com/vmaravind96/NetflixMovieRecommendation)[Movie](https://github.com/vmaravind96/NetflixMovieRecommendation)[ ](https://github.com/vmaravind96/NetflixMovieRecommendation)[Recommendation](https://github.com/vmaravind96/NetflixMovieRecommendation)





Netflix Movie Recommendation System

11

REFERENCES

[1] [Yehuda](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[Koren.](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[2010.](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[Collaborative](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ﬁltering](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[with](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[temporal](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[dynamics.](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[Commun.](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ACM](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[53,](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[4](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[(April](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[2010),](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[ ](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)[89–97.](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)

[DOI:https://doi.org/10.1145/1721654.1721677](https://www.cc.gatech.edu/~zha/CSE8801/CF/kdd-fp074-koren.pdf)

[2] [Y.](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Koren,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[R.](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Bell](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[and](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[C.](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Volinsky,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)["Matrix](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Factorization](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Techniques](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[for](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Recommender](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Systems,"](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[in](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Computer,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[vol.](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[42,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[no.](https://ieeexplore.ieee.org/document/5197422)

[8,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[pp.](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[30-37,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[Aug.](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[2009,](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[doi:](https://ieeexplore.ieee.org/document/5197422)[ ](https://ieeexplore.ieee.org/document/5197422)[10.1109/MC.2009.263.](https://ieeexplore.ieee.org/document/5197422)

[3] [R.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[M.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[Bell,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[J.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[Bennett,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[Y.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[Koren](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[and](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[C.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[Volinsky,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)["The](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[million](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[dollar](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[programming](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[prize,"](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[in](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[IEEE](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[Spectrum,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[vol.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[46,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[no.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[5,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

[pp.](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[28-33,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[May](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[2009,](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[doi:](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[ ](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)[10.1109/MSPEC.2009.4907383](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)

[4] [P.](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Bedi,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[C.](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Sharma,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[P.](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Vashisth,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[D.](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Goel](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[and](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[M.](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Dhanda,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)["Handling](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[cold](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[start](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[problem](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[in](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Recommender](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Systems](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[by](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[using](https://ieeexplore.ieee.org/document/7275909)

[Interaction](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Based](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Social](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Proximity](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[factor,"](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[2015](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[International](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Conference](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[on](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Advances](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[in](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Computing,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Communications](https://ieeexplore.ieee.org/document/7275909)

[and](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[Informatics](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[(ICACCI),](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[2015,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[pp.](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[1987-1993,](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[doi:](https://ieeexplore.ieee.org/document/7275909)[ ](https://ieeexplore.ieee.org/document/7275909)[10.1109/ICACCI.2015.7275909.](https://ieeexplore.ieee.org/document/7275909)

[5] [A](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[ ](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[Beginner’s](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[ ](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[guide](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[ ](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[to](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[ ](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)[XGBoost](https://towardsdatascience.com/a-beginners-guide-to-xgboost-87f5d4c30ed7)

[6] [Matrix](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))[ ](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))[factorization](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))[ ](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))[(recommender](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))[ ](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))[systems)](https://en.wikipedia.org/wiki/Matrix_factorization_\(recommender_systems\))

[7] [Machine](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Learning](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[-](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Singular](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Value](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Decomposition](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[(SVD)](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[and](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Principal](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Component](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[Analysis](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[ ](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)[(PCA)](https://jonathan-hui.medium.com/machine-learning-singular-value-decomposition-svd-principal-component-analysis-pca-1d45e885e491)

[8] [Building](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[and](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[Testing](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[Recommender](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[Systems](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[With](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[Surprise,](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[ ](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)[Step-By-Step](https://towardsdatascience.com/building-and-testing-recommender-systems-with-surprise-step-by-step-d4ba702ef80b)


