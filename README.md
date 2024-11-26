import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from mpl_toolkits.axes_grid1
import make_axes_locatable
from sklearn.cluster 
import KMeans
from sklearn.metrics 
import mean_squared_error
import itertools
from sklearn.metrics
import silhouette_samples, silhouette_score
from scipy.sparse 
import csr_matrix
movies = pd.read_csv('Movie.csv')
ratings = pd.read_csv('Rate.csv')
dataset = pd.merge(movies, ratings, how ='inner', on ='movieId')
dataset.head()
print('The dataset contains: ', len(ratings), ' ratings of ', len(movies), ' movies.')
dataset.shape
dataset.nunique()
unique_user = ratings.userId.nunique(dropna = True)
unique_movie = ratings.movieId.nunique(dropna = True)
print("number of unique user:")
print(unique_user)
print("number of unique movies:")
print(unique_movie)
dataset = dataset.drop_duplicates()
print(dataset)
dataset.describe()
dataset.isnull()
dataset.isnull().sum()
x = dataset.genres
a = list()
for i in x:
 abc = i
 a.append(abc.split('|'))
a = pd.DataFrame(a) 
b = a[0].unique()
for i in b:
 dataset[i] = 0
dataset.head(2000)
for i in b:
 dataset.loc[dataset['genres'].str.contains(i), i] = 1
dataset.head(2000)
dataset = dataset.drop(['genres','title'],axis =1)
dataset.head()
a=dataset
a=a.groupby('movieId')["rating"].mean()
sorted_ratings_wise_movie=a.sort_values(ascending=False)
sorted_ratings_wise_movie
def get_genre_ratings(ratings, movies, genres, column_names):
 genre_ratings = pd.DataFrame()
 for genre in genres: 
 genre_movies = movies[movies['genres'].str.contains(genre) ]
 avg_genre_votes_per_user = ratings[ratings['movieId'].isin(genre_movies['movieId'])].loc[:, 
['userId', 'rating']].groupby(['userId'])['rating'].mean().round(2)
 
 genre_ratings = pd.concat([genre_ratings, avg_genre_votes_per_user], axis=1)
 
 print(genre_ratings)
 genre_ratings.columns = column_names
 return genre_ratings
genre_ratings = get_genre_ratings(ratings, movies, ['Romance', 'Sci-Fi', 'Comedy'], 
['avg_romance_rating', 'avg_scifi_rating', 'avg_comedy_rating'])
genre_ratings.head()
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
 biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & 
(genre_ratings['avg_scifi_rating'] > score_limit_2)) | ((genre_ratings['avg_scifi_rating'] < 
score_limit_1) & (genre_ratings['avg_romance_rating'] > score_limit_2))]
 biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
 biased_dataset = pd.DataFrame(biased_dataset.to_records())
 return biased_dataset
biased_dataset = bias_genre_rating_dataset(genre_ratings, 3.2, 2.5)
print( "Number of records: ", len(biased_dataset))
biased_dataset.head()
X = biased_dataset[['avg_scifi_rating','avg_romance_rating','avg_comedy_rating']].values
df = biased_dataset[['avg_scifi_rating','avg_romance_rating','avg_comedy_rating']]
possible_k_values = range(2, len(X)+1, 5)
def clustering_errors(k, data):
k-means = KMeans(n_clusters=k).fit(data)
predictions = kmeans.predict(data)
 #cluster_centers = kmeans.cluster_centers_
 # errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data.values,predictions)
 # return sum(errors)
silhouette_avg = silhouette_score(data, predictions)
return silhouette_avg
errors_per_k = [clustering_errors(k, X) for k in possible_k_values]
fig, ax = plt.subplots(figsize=(16, 6))
plt.plot(possible_k_values, errors_per_k)
xticks = np.arange(min(possible_k_values), max(possible_k_values)+1, 5.0)
ax.set_xticks(xticks, minor=False)
ax.set_xticks(xticks, minor=True)
ax.xaxis.grid(True, which='both')
yticks = np.arange(round(min(errors_per_k), 2), max(errors_per_k), .05)
ax.set_yticks(yticks, minor=False)
ax.set_yticks(yticks, minor=True)
ax.yaxis.grid(True, which='both')
ratings_title = pd.merge(ratings, movies[['movieId', 'title']], on='movieId' )
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
print('dataset dimensions: ', user_movie_ratings.shape, '\n\nSubset example:')
user_movie_ratings.iloc[:6, :10]
def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
 most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
 most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
 return most_rated_movies
def get_most_rated_movies(user_movie_ratings, max_number_of_movies):
 # 1- Count
 user_movie_ratings = user_movie_ratings.append(user_movie_ratings.count(), ignore_index=True)
 # 2- sort
 user_movie_ratings_sorted = user_movie_ratings.sort_values(len(user_movie_ratings)-1, axis=1, ascending=False)
 user_movie_ratings_sorted = user_movie_ratings_sorted.drop(user_movie_ratings_sorted.tail(1).index)
 # 3- slice
 most_rated_movies = user_movie_ratings_sorted.iloc[:, :max_number_of_movies]
 return most_rated_movies
def get_users_who_rate_the_most(most_rated_movies, max_number_of_movies):
 # Get most voting users
 # 1- Count
 most_rated_movies['counts'] = pd.Series(most_rated_movies.count(axis=1))
 # 2- Sort
 most_rated_movies_users = most_rated_movies.sort_values('counts', ascending=False)
 # 3- Slice
 most_rated_movies_users_selection = most_rated_movies_users.iloc[:max_number_of_movies, :]
 most_rated_movies_users_selection = most_rated_movies_users_selection.drop(['counts'], axis=1)
 
 return most_rated_movies_users_selection
n_movies = 30
n_users = 18
most_rated_movies_users_selection = sort_by_rating_density(user_movie_ratings, n_movies, 
n_users)
print('dataset dimensions: ', most_rated_movies_users_selection.shape)
most_rated_movies_users_selection.head()
def draw_movies_heatmap(most_rated_movies_users_selection, axis_labels=True):
 
 # Reverse to match the order of the printed dataframe
 #most_rated_movies_users_selection = most_rated_movies_users_selection.iloc[::-1]
 fig = plt.figure(figsize=(15,4))
 ax = plt.gca()
 
# Draw heatmap
 heatmap = ax.imshow(most_rated_movies_users_selection, interpolation='nearest', vmin=0, 
vmax=5, aspect='auto')
 if axis_labels:
 ax.set_yticks(np.arange(most_rated_movies_users_selection.shape[0]) , minor=False)
 ax.set_xticks(np.arange(most_rated_movies_users_selection.shape[1]) , minor=False)
 ax.invert_yaxis()
 ax.xaxis.tick_top()
 labels = most_rated_movies_users_selection.columns.str[:40]
 ax.set_xticklabels(labels, minor=False)
 ax.set_yticklabels(most_rated_movies_users_selection.index, minor=False)
 plt.setp(ax.get_xticklabels(), rotation=90)
 else:
 ax.get_xaxis().set_visible(False)
 ax.get_yaxis().set_visible(False)
 
 ax.grid(False)
 ax.set_ylabel('User id')
 # Separate heatmap from color bar
 divider = make_axes_locatable(ax)
 cax = divider.append_axes("right", size="5%", pad=0.05)
 # Color bar
 cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
 cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])
 plt.show()
 draw_movies_heatmap(most_rated_movies_users_selection)
user_movie_ratings = pd.pivot_table(ratings_title, index='userId', columns= 'title', values='rating')
most_rated_movies_1k = get_most_rated_movies(user_movie_ratings, 1000)
def sparse_clustering_errors(k, data):
 kmeans = KMeans(n_clusters=k).fit(data)
 predictions = kmeans.predict(data)
 cluster_centers = kmeans.cluster_centers_
 errors = [mean_squared_error(row, cluster_centers[cluster]) for row, cluster in zip(data, 
predictions)]
 return sum(errors)
sparse_ratings = csr_matrix(pd.SparseDataFrame(most_rated_movies_1k).to_coo())
def draw_movie_clusters(clustered, max_users, max_movies):
 c=1
 for cluster_id in clustered.group.unique():
 # To improve visibility, we're showing at most max_users users and max_movies movies per 
cluster.
 # You can change these values to see more users & movies per cluster
d = clustered[clustered.group == cluster_id].drop(['index', 'group'], axis=1)
n_users_in_cluster = d.shape[0]
 
d = sort_by_rating_density(d, max_movies, max_users)
 
d = d.reindex_axis(d.mean().sort_values(ascending=False).index, axis=1)
d = d.reindex_axis(d.count(axis=1).sort_values(ascending=False).index)
d = d.iloc[:max_users, :max_movies]
n_users_in_plot = d.shape[0]
 
 # We're only selecting to show clusters that have more than 9 users, otherwise, they're less 
interesting
if len(d) > 9:
 print('cluster # {}'.format(cluster_id))
 print('# of users in cluster: {}.'.format(n_users_in_cluster), 
# of users in plot: 
{}'.format(n_users_in_plot))
 fig = plt.figure(figsize=(15,4))
 ax = plt.gca()
 ax.invert_yaxis()
 ax.xaxis.tick_top()
 labels = d.columns.str[:40]
 ax.set_yticks(np.arange(d.shape[0]) , minor=False)
 ax.set_xticks(np.arange(d.shape[1]) , minor=False)
 ax.set_xticklabels(labels, minor=False)
 
 ax.get_yaxis().set_visible(False)
 # Heatmap
 heatmap = plt.imshow(d, vmin=0, vmax=5, aspect='auto')
 ax.set_xlabel('movies')
 ax.set_ylabel('User id')
 divider = make_axes_locatable(ax)
 cax = divider.append_axes("right", size="5%", pad=0.05)
 # Color bar
 cbar = fig.colorbar(heatmap, ticks=[5, 4, 3, 2, 1, 0], cax=cax)
 cbar.ax.set_yticklabels(['5 stars', '4 stars','3 stars','2 stars','1 stars','0 stars'])
 plt.setp(ax.get_xticklabels(), rotation=90, fontsize=9)
 plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', labelbottom='off', 
labelleft='off') 
 #print('cluster # {} \n(Showing at most {} users and {} movies)'.format(cluster_id, max_users, 
max_movies))
 plt.show()
def bias_genre_rating_dataset(genre_ratings, score_limit_1, score_limit_2):
 biased_dataset = genre_ratings[((genre_ratings['avg_romance_rating'] < score_limit_1 - 0.2) & 
(genre_ratings['avg_scifi_rating'] > score_limit_2)) | ((genre_ratings['avg_scifi_rating'] < 
score_limit_1) & (genre_ratings['avg_romance_rating'] > score_limit_2))]
 biased_dataset = pd.concat([biased_dataset[:300], genre_ratings[:2]])
 biased_dataset = pd.DataFrame(biased_dataset.to_records())
 return biased_dataset
def sort_by_rating_density(user_movie_ratings, n_movies, n_users):
 most_rated_movies = get_most_rated_movies(user_movie_ratings, n_movies)
 most_rated_movies = get_users_who_rate_the_most(most_rated_movies, n_users)
 return most_rated_movies
import helper
import importlib
importlib.reload(helper)
predictions = KMeans(n_clusters=20, algorithm='full').fit_predict(sparse_ratings)
max_users = 70
max_movies = 50
clustered = pd.concat([most_rated_movies_1k.reset_index(), pd.DataFrame({'group':predictions})], 
axis=1)
helper.draw_movie_clusters(clustered, max_users, max_movies)
cluster_number = 4
n_users = 75
n_movies = 300
cluster = clustered[clustered.group == cluster_number].drop(['index', 'group'], axis=1)
cluster = sort_by_rating_density(cluster, n_movies, n_users)
draw_movies_heatmap(cluster, axis_labels=False)
cluster.fillna('').head()
movie_name = ("Blues Brothers, The (1980)")
cluster[movie_name].mean()
cluster.mean().head(20)
user_id = 2
user_2_ratings = cluster.loc[user_id, :]
user_2_unrated_movies = user_2_ratings[user_2_ratings.isnull()]
avg_ratings = pd.concat([user_2_unrated_movies, cluster.mean()], axis=1, join='inner').loc[:,0]
avg_ratings.sort_values(ascending=False)[:20]
from sklearn.neighbors import NearestNeighbors
import seaborn as sns
dataset = ratings.pivot(index='movieId',columns='userId',values='rating')
dataset.head()
dataset.fillna(0,inplace=True)
dataset.head()
no_user_voted = ratings.groupby('movieId')['rating'].agg('count')
no_movies_voted = ratings.groupby('userId')['rating'].agg('count')
f,ax = plt.subplots(1,1,figsize=(16,4))
# ratings['rating'].plot(kind='hist')
plt.scatter(no_user_voted.index,no_user_voted,color='mediumseagreen')
plt.axhline(y=10,color='r')
plt.xlabel('MovieId')
plt.ylabel('No. of users voted')
plt.show()
dataset = dataset.loc[no_user_voted[no_user_voted > 10].index,:]
f,ax = plt.subplots(1,1,figsize=(16,4))
plt.scatter(no_movies_voted.index,no_movies_voted,color='mediumseagreen')
plt.axhline(y=50,color='r')
plt.xlabel('UserId')
plt.ylabel('No. of votes by user')
plt.show()
dataset=dataset.loc[:,no_movies_voted[no_movies_voted > 50].index]
dataset
sample = np.array([[0,0,3,0,0],[4,0,0,0,2],[0,0,0,0,1]])
sparsity = 1.0 - ( np.count_nonzero(sample) / float(sample.size) )
print(sparsity)
csr_sample = csr_matrix(sample)
print(csr_sample)
csr_data = csr_matrix(dataset.values)
dataset.reset_index(inplace=True)
knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
knn.fit(csr_data)
def get_movie_recommendation(movie_name):
n_movies_to_reccomend = 20
movie_list = movies[movies['title'].str.contains(movie_name)] 
if len(movie_list): 
movie_idx = movie_list.iloc[0]['movieId']
movie_idx = dataset[dataset['movieId'] == movie_idx].index[0]
distances , indices = knn.kneighbors(csr_data[movie_idx],n_neighbors=n_movies_to_reccomend+1) 
rec_movie_indices = sorted(list(zip(indices.squeeze().tolist(),distances.squeeze().tolist())),key=lambda x: x[1])[:0:-1]
recommend_frame = []
for val in rec_movie_indices:
movie_idx = dataset.iloc[val[0]]['movieId']
idx = movies[movies['movieId'] == movie_idx].index
recommend_frame.append({'Title':movies.iloc[idx]['title'].values[0],'Distance':val[1]})
df = pd.DataFrame(recommend_frame,index=range(1,n_movies_to_reccomend+1))
return df
else:
return "No movies found. Please check your input"
get_movie_recommendation('Blues Brothers, The')
import numpy as np
import pickle
from sklearn.cluster import AffinityPropagation
import math
import sys
import pickle
from time import sleep
train_data=[]
test_data = []
users ={}
movies={}
def get_data():
 train_data = np.genfromtxt("dataset.csv",delimiter= ',',skip_header=(1))
 
 user_id = list(set(train_data[:,0]))
 user_id.sort()
 movie_id = list(set(train_data[:,1]))
 movie_id.sort()
 users={}
 movies={}
 for i,j in enumerate(user_id):
 users[j]=i
 for i,j in enumerate(movie_id):
 
# print(i)
 movies[j]=i
 
 
 user_item = np.empty((len(set(train_data[:,0])),len(set(train_data[:,1]))))
 
 for row in train_data:
 i = users[int(row[0])]
 j = movies[int(row[1])] 
 user_item[i][j]=row[2]
 return(train_data,users,movies,user_item)
 
 
def find_similarity(): 
 movie_sim = np.zeros([len(movies.keys()),len(movies.keys())])
 for st,m1 in enumerate(movies.keys()):
 if st%1000==0:
 print('in movie',st) 
 
# r1 - Average rating of movie m1
 r1 = np.average(user_item[:,movies[m1]])
 u_m1 = np.where(user_item[:,movies[m1]]!=0)
 
 for j in range(st,len(movies.keys())):
     m2 = 1
 myArr = list(movies.keys())
 m2 =myArr[j]
 r2 = np.average(user_item[:,movies[m2]]) 
 u_m2 = np.where(user_item[:,movies[m2]]!=0)
 u = list(set(u_m1[0]).intersection(set(u_m2[0]))) 
 if len(u)!=0:
 co_ratings = user_item[np.ix_(u,[int(movies[m1]),int(movies[m2])])]
 num = sum((co_ratings[:,0]-r1)*(co_ratings[:,1]-r2))
 den = ((sum((co_ratings[:,0]-r1)*2))0.5)((sum((co_ratings[:,1]-r2)*2))*0.5)
 corr = num*1.0/den
 movie_sim[st][j] = corr
 if j != st:
 movie_sim[j][st] = corr
 
 
 return(movie_sim) 
 
def compute_reco(act_user,act_mov):
 
 user = users[act_user]
 movie = movies[act_mov]
 clus = clus_labels[movie]
 clus_movie = np.where(clus_labels==clus)
 user_rated_movies = np.where(user_item[user]!=0)
 rated_movies = list(set(clus_movie[0]).intersection(set(user_rated_movies[0]))) 
 if movie in rated_movies:
 rated_movies.remove(movie)
 clus_movie = np.delete(clus_movie,np.where(clus_movie[0]==movie))
 ratings = user_item[user,rated_movies]
 # dtype = [('movie_num', int), ('rating', float), ('W', int)]
 reco_ratings = np.zeros([len(clus_movie),3])
 for j,m in enumerate(clus_movie):
 if m in user_rated_movies[0]:
 pred_rating = user_item[user,m]
 reco_ratings[j]=[0,m,pred_rating]
 else: 
 sim = sim_mat[m,rated_movies]
 rated = np.column_stack((ratings,sim))
 pred_rating = np.dot(rated[:,0],rated[:,1])*1.0/sum(rated[:,1])
 pred_rating = round(pred_rating * 2) / 2 
 if(math.isnan(pred_rating)):
 pred_rating = 0 
 reco_ratings[j]=[1,m,pred_rating]
 not_watched_ind = np.where(reco_ratings[:,0]==1)
 if(len(not_watched_ind[0])>10):
 not_watched = reco_ratings[not_watched_ind[0]]
 reco_movies = not_watched[np.argsort(not_watched[:, 2])][::-1]
 reco_movies= reco_movies[0:10]
 elif (len(reco_ratings)>10):
 reco_movies = reco_ratings[np.argsort(reco_ratings[:, 2])][::-1]
 reco_movies= reco_movies[0:10]
 else:
 reco_movies = reco_ratings[np.argsort(reco_ratings[:, 2])][::-1]
 final_list = [0]*len(reco_movies)
 for k,i in enumerate(reco_movies):
 final_list[k] = next(key for key, value in movies.iteritems() if value == i[1] )
 return(final_list)
 
prompt = "Do you want to compute similarity matrix and cluster?\n Enter Y - To compute the 
components \n Enter N - To use precomputed components\n Enter ex to stop execution \nInput - "
while True:
    inp_choice = input(prompt)
 if inp_choice.lower() == 'y':
 train_data,users,movies,user_item = get_data()
 print("Data read complete\n Computing similarity...")
 sleep(6)
 print("computation completed")
 sleep(3)
 print("Results are processing")
 break
 
 
 sim_mat = find_similarity() 
 print("Similarity matrix computed\n Movies being clustered...")
 np.savetxt("sim_mat_Pearson.csv",sim_mat,delimiter=',') 
 
 af = AffinityPropagation(verbose=True,affinity="precomputed").fit(sim_mat) 
 clus_labels = af.labels_ 
 print("Movies clustered")
 
 break
 elif inp_choice.lower()=='n':
 print("Loading precomputed components...")
 data2 = []
 with open("Pickle_file", "rb") as f:
 for _ in range(pickle.load(f)):
 data2.append(pickle.load(f)) 
 sim_mat = np.genfromtxt("sim_mat_Pearson.csv",delimiter=",") 
 train_data = data2[0]
 # test_data = data2[1]
 users = data2[1]
 movies = data2[2]
 user_item = data2[3]
 clus_labels = data2[4]
 print("\nPrecomputed components loaded")
 break
 elif inp_choice.lower()=="ex":
 sys.exit("Program stopped as requested")
 else:
 print("Invalid input")
 continue
 
while True:
 break
 print("\n1st 100 User ids =", users.keys()[0:100],)
 print("\n")
 print("1st 100 Movie ids =",movies.keys()[0:100],)
 break 
 print("hello1")
# 
 inp = input("\nEnter user id and movie id separated by comma- ") 
 if inp == "": 
 break 
 else:
 act_user,act_mov = inp.split(',')
 act_user = int(act_user)
 act_mov = int(act_mov)
 final_list = compute_reco(act_user,act_mov)
 print("Top %d movies for user %d similar to movie %d \n"%(len(final_list),act_user,act_mov))
 print(final_list)
 inp1 = input("Do you want to continue? Y/N - ")
 if inp1.lower()== 'y':
 continue
 else:
 break 
computeRecommedationMatrix()
