from flask import *
import os
import numpy as np 
import pandas as pd 
from scipy import spatial
import warnings
warnings.filterwarnings('ignore')
import re
import os
import csv




app=Flask(__name__)
@app.route('/')
def index():
    return render_template("index.html")

@app.route('/title_home')
def title_home():
    return render_template("title_home.html")

@app.route('/logic',methods=['POST'])
def logic():
    if request.method=="POST":
        #ratings = pd.read_csv("ratings_final1.csv")
        mid=request.form['text']
        #gen=request.form['text']
        #print(mid,type(mid),"haiii")
        ratings = pd.read_csv("ratings_final1.csv")
        # links = pd.read_csv("../input/links.csv")
        tags = pd.read_csv("tags_final.csv")
        movies = pd.read_csv("final_movies.csv")

        data={}
        with open("final_movies.csv") as f1:
            records=list(csv.reader(f1))
            for row in records:
                data[row[1]]=row[0]
        mid=int(data[mid])


        userRatingsAggr = ratings.groupby(['userId']).agg({'rating': [np.size, np.mean]})
        userRatingsAggr.reset_index(inplace=True)  # To reset multilevel (pivot-like) index
        

        movieRatingsAggr = ratings.groupby(['movieId']).agg({'rating': [np.size, np.mean]})
        movieRatingsAggr.reset_index(inplace=True)
       

        movies = movies.merge(movieRatingsAggr, left_on='movieId', right_on='movieId', how='left')  # ['rating']
        movies.columns = ['movieId', 'title', 'genres', 'rating_count', 'rating_avg']


        # In[ ]:


        movies.head(5)


        # ### Get movie years from title

        # In[ ]:


        def getYear(title):
            result = re.search(r'\(\d{4}\)', title)
            if result:
                found = result.group(0).strip('(').strip(')')
            else: 
                found = 0
            return int(found)
            
        movies['year'] = movies.apply(lambda x: getYear(x['title']), axis=1)
        # movies.head(10)


        # ### Create genres matrix - one hot encoding
        # At this step I create a "genresMatrix" field where every value is a list of binary values (19 elements in every list, for the 19 possible genres).  For example a movie with genres "Action", "Adventure" and "Children" will look like: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] . This transformation called as "one hot encoding".
        # 
        # This matrix will be very useful to define the similarities between two "genres" sets. For this purpose I'm going to compute the Cosine distance between the given arrays. More info: [SciPy spatial.distance.cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)

        # In[ ]:


        genresList = [
          "Action",
          "Adventure",
          "Animation",
          "Children",
          "Comedy",
          "Crime",
          "Documentary",
          "Drama",
          "Fantasy",
          "Film-Noir",
          "Horror",
          "Musical",
          "Mystery",
          "Romance",
          "Sci-Fi",
          "Thriller",
          "War",
          "Western",
          "(no genres listed)"
        ]
        def getmovie(genres):
            movieGenresList = genres.split('|')
            return movieGenresList
        def setGenresMatrix(genres):
            movieGenresMatrix = []
            movieGenresList = genres.split('|')
            for x in genresList:
                if (x in movieGenresList):
                    movieGenresMatrix.append(1)
                else:
                    movieGenresMatrix.append(0) 
            return movieGenresMatrix
        #def gettitle(genre):
        #    for i in moviesWithSimGenre.values:
         #       if genre in i['update_genres']:
          #          return i['title']
            
        movies['genresMatrix'] = movies.apply(lambda x: np.array(list(setGenresMatrix(x['genres']))), axis=1)
        movies['update_genres']= movies.apply(lambda x: np.array(list(getmovie(x['genres']))), axis=1)


        def setRatingGroup(numberOfRatings):
            # if (numberOfRatings is None): return 0
            if (1 <= numberOfRatings <= 10): return 1
            elif (11 <= numberOfRatings <= 30): return 2
            elif (31 <= numberOfRatings <= 100): return 3
            elif (101 <= numberOfRatings <= 300): return 4
            elif (301 <= numberOfRatings <= 1000): return 5
            elif (1001 <= numberOfRatings): return 6
            else: return 0

        movies['ratingGroup'] = movies.apply(lambda x: setRatingGroup(x['rating_count']), axis=1)
        movies.fillna(0, inplace=True)  # Replace NaN values to zero

        # movies.head(10)


        # ### Tags 
        # Iterate through all the user given tags, split the tags into words, filter the defined stop words (frequent English words) and put the results into a dictionary that indexed by movieId.

        # In[ ]:


        stopWords = ['a', 'about', 'above', 'above', 'across', 'after', 'afterwards', 'again', 'against', 'all', 'almost', 
                'alone', 'along', 'already', 'also','although','always','am','among', 'amongst', 'amoungst', 'amount',  'an', 'and', 
                'another', 'any','anyhow','anyone','anything','anyway', 'anywhere', 'are', 'around', 'as',  'at', 'back','be','became', 
                'because','become','becomes', 'becoming', 'been', 'before', 'beforehand', 'behind', 'being', 'below', 'beside', 'besides', 
                'between', 'beyond', 'bill', 'both', 'bottom','but', 'by', 'call', 'can', 'cannot', 'cant', 'co', 'con', 'could', 'couldnt', 
                'cry', 'de', 'describe', 'detail', 'do', 'done', 'down', 'due', 'during', 'each', 'eg', 'eight', 'either', 'eleven','else', 
                'elsewhere', 'empty', 'enough', 'etc', 'even', 'ever', 'every', 'everyone', 'everything', 'everywhere', 'except', 'few', 'fifteen', 
                'fify', 'fill', 'find', 'fire', 'first', 'five', 'for', 'former', 'formerly', 'forty', 'found', 'four', 'from', 'front', 'full', 
                'further', 'get', 'give', 'go', 'had', 'has', 'hasnt', 'have', 'he', 'hence', 'her', 'here', 'hereafter', 'hereby', 'herein', 'hereupon', 
                'hers', 'herself', 'him', 'himself', 'his', 'how', 'however', 'hundred', 'ie', 'if', 'in', 'inc', 'indeed', 'interest', 'into', 'is', 'it', 
                'its', 'itself', 'keep', 'last', 'latter', 'latterly', 'least', 'less', 'ltd', 'made', 'many', 'may', 'me', 'meanwhile', 'might', 'mill', 
                'mine', 'more', 'moreover', 'most', 'mostly', 'move', 'much', 'must', 'my', 'myself', 'name', 'namely', 'neither', 'never', 'nevertheless', 
                'next', 'nine', 'no', 'nobody', 'none', 'noone', 'nor', 'not', 'nothing', 'now', 'nowhere', 'of', 'off', 'often', 'on', 'once', 'one', 
                'only', 'onto', 'or', 'other', 'others', 'otherwise', 'our', 'ours', 'ourselves', 'out', 'over', 'own','part', 'per', 'perhaps', 'please', 
                'put', 'rather', 're', 'same', 'see', 'seem', 'seemed', 'seeming', 'seems', 'serious', 'several', 'she', 'should', 'show', 'side', 'since', 
                'sincere', 'six', 'sixty', 'so', 'some', 'somehow', 'someone', 'something', 'sometime', 'sometimes', 'somewhere', 'still', 'such', 'system', 
                'take', 'ten', 'than', 'that', 'the', 'their', 'them', 'themselves', 'then', 'thence', 'there', 'thereafter', 'thereby', 'therefore', 'therein', 
                'thereupon', 'these', 'they', 'thickv', 'thin', 'third', 'this', 'those', 'though', 'three', 'through', 'throughout', 'thru', 'thus', 'to', 
                'together', 'too', 'top', 'toward', 'towards', 'twelve', 'twenty', 'two', 'un', 'under', 'until', 'up', 'upon', 'us', 'very', 'via', 'was', 
                'we', 'well', 'were', 'what', 'whatever', 'when', 'whence', 'whenever', 'where', 'whereafter', 'whereas', 'whereby', 'wherein', 'whereupon', 
                'wherever', 'whether', 'which', 'while', 'whither', 'who', 'whoever', 'whole', 'whom', 'whose', 'why', 'will', 'with', 'within', 'without', 'would', 
                'yet', 'you', 'your', 'yours', 'yourself', 'yourselves', 'the']

        tagsDict = {}

        for index, x in tags.iterrows():
            wordlist = str(x['tag']).lower().split(' ')
            movieId = x['movieId']
            for y in wordlist:
                if y not in stopWords:
                    if movieId in tagsDict:
                        # if y not in tagsDict[movieId]:  # Switched off (we will get a non unique list)
                            tagsDict[movieId].append(y)
                    else:
                        tagsDict[movieId] = [y]


        titleWordsDict = {}

        for index, x in movies.iterrows():
            wordlist = str(x['title']).lower().split(' ')
            movieId = x['movieId']
            for y in wordlist:
                if y not in stopWords:
                    if movieId in titleWordsDict:
                            titleWordsDict[movieId].append(y)
                    else:
                        titleWordsDict[movieId] = [y]


        # # Movie recommendation algorithm
        # 
        # The algorithm gets a "movieId" as input parameter and computes the similarity for every other movie in the dataset. To fine-tuning this process, we can set up weights for the 6 defined similarity attributes at the beginning of the code. At the end we just ordering the result set (the most similar movies will be at the beginning). 

        # In[ ]:


        # Parameter weights
        genresSimilarityWeight = 0.8
        tagsSimilarityWeight = 2
        titleSimilarityWeight = 1
        ratingAvgWeight = 0.2
        ratingGroupWeight = 0.005
        yearDistanceWeight = 0.1

        def tagsSimilarity(basisMovieID, checkedMovieID, checkType):    
            # The higher value is the more similar (from 0 to 1) 
            if checkType == 'tag':
                dictToCheck = tagsDict
            else:
                dictToCheck = titleWordsDict
                
            counter = 0
            if basisMovieID in dictToCheck: 
                basisTags = dictToCheck[basisMovieID]
                countAllTags = len(basisTags)
                basisTagsDict = {}
                for x in basisTags:
                    if x in basisTagsDict:
                        basisTagsDict[x] += 1
                    else:
                        basisTagsDict[x] = 1   
                
                for x in basisTagsDict:
                    basisTagsDict[x] = basisTagsDict[x] / countAllTags
            else: return 0
            
            if checkedMovieID in dictToCheck: 
                checkedTags = dictToCheck[checkedMovieID]
                checkedTags = set(checkedTags) # Make the list unique
                checkedTags = list(checkedTags)
                
            else: return 0
            
            for x in basisTagsDict:
                if x in checkedTags: counter += basisTagsDict[x]
            return counter
        #def getmovie(genre):
            #for i in movies['genresMatrix']:
        #def genreSimilarity(genre):
         #   moviesWithSimGenre=movies
          #  l=[]
           # for i in moviesWithSimGenre.values:
            #    g=[]
             #   if genre in i[7]:
              #      g.append(i[1])
               #     g.append(i[4])
                #    l.append(g)
                #print(i[7])
            #print(l)
            #return l
        def checkSimilarity(movieId):
            # print("SIMILAR MOVIES TO:")
            # print (movies[movies['movieId'] == movieId][['title', 'rating_count', 'rating_avg']])
            #movies['genresMatrix'] = movies.apply(lambda x: np.array(list(movies[movies['movieId'] == movieId](setGenresMatrix(x['genres'])))), axis=1)
            basisGenres = np.array(list(movies[movies['movieId'] == movieId]['genresMatrix']))
            #print(movies['genresMatrix']["Action"])
            basisYear = int(movies[movies['movieId'] == movieId]['year'])
            basisRatingAvg = movies[movies['movieId'] == movieId]['rating_avg']
            basisRatingGroup = movies[movies['movieId'] == movieId]['ratingGroup']
            
            moviesWithSim = movies
            moviesWithSim['similarity'] = moviesWithSim.apply(lambda x: 
                                                              spatial.distance.cosine(x['genresMatrix'], basisGenres) * genresSimilarityWeight + 
                                                              - tagsSimilarity(movieId, x['movieId'], 'tag') * tagsSimilarityWeight +
                                                              - tagsSimilarity(movieId, x['movieId'], 'title') * titleSimilarityWeight +
                                                              abs(basisRatingAvg - x['rating_avg']) * ratingAvgWeight +
                                                              abs(basisRatingGroup - x['ratingGroup']) * ratingGroupWeight + 
                                                              abs(basisYear - x['year'])/100 * yearDistanceWeight
                                                             , axis=1)
            x=moviesWithSim.loc[(moviesWithSim.movieId == movieId)]
            moviesWithSim = moviesWithSim.loc[(moviesWithSim.movieId != movieId)]
            #return moviesWithSim[['title', 'genres','rating_avg', 'similarity']].sort_values('similarity')
            M = moviesWithSim[['title', 'genres','rating_avg', 'similarity']].sort_values('similarity')
            return M[['title', 'genres','rating_avg']],x[['title', 'genres','rating_avg']]


        # In[ ]:


        # currentMovie = movies.loc[(movies.movieId == 3793)]
        # currentMovie.head(1)


        # # Movie recommendations

        # <img style="float: left;" src="https://movieposters2.com/images/1125336-b.jpg" width="280" height="400"> 
        # <span style="font-size:200%;margin:20px;">X-men</span>

        # In[ ]:


        # X-men
        #similarityResult=[]
        #t=genreSimilarity(gen)
        similarityResult,e = checkSimilarity(mid)
        res=similarityResult.head(5).values
        r=e.values
           
    return render_template("title_home.html",result=res,result1=r)
@app.route('/genre_home')
def genre_home():
    return render_template("genre_home.html")

@app.route('/genre_logic',methods=['POST'])
def genre_logic():
    if request.method=="POST":
        #ratings = pd.read_csv("ratings_final1.csv")
        #mid=request.form['text']
        gen=request.form['text']
        #print(mid,type(mid),"haiii")
        ratings = pd.read_csv("ratings_final1.csv")
        # links = pd.read_csv("../input/links.csv")
        tags = pd.read_csv("tags_final.csv")
        movies = pd.read_csv("final_movies.csv")

        data={}
        with open("final_movies.csv") as f1:
            records=list(csv.reader(f1))
            for row in records:
                data[row[1]]=row[0]
        #mid=int(data[mid])


        userRatingsAggr = ratings.groupby(['userId']).agg({'rating': [np.size, np.mean]})
        userRatingsAggr.reset_index(inplace=True)  # To reset multilevel (pivot-like) index
        

        movieRatingsAggr = ratings.groupby(['movieId']).agg({'rating': [np.size, np.mean]})
        movieRatingsAggr.reset_index(inplace=True)
       

        movies = movies.merge(movieRatingsAggr, left_on='movieId', right_on='movieId', how='left')  # ['rating']
        movies.columns = ['movieId', 'title', 'genres', 'rating_count', 'rating_avg']


        # In[ ]:


        movies.head(5)


        # ### Get movie years from title

        # In[ ]:


        def getYear(title):
            result = re.search(r'\(\d{4}\)', title)
            if result:
                found = result.group(0).strip('(').strip(')')
            else: 
                found = 0
            return int(found)
            
        movies['year'] = movies.apply(lambda x: getYear(x['title']), axis=1)
        # movies.head(10)


        # ### Create genres matrix - one hot encoding
        # At this step I create a "genresMatrix" field where every value is a list of binary values (19 elements in every list, for the 19 possible genres).  For example a movie with genres "Action", "Adventure" and "Children" will look like: [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] . This transformation called as "one hot encoding".
        # 
        # This matrix will be very useful to define the similarities between two "genres" sets. For this purpose I'm going to compute the Cosine distance between the given arrays. More info: [SciPy spatial.distance.cosine](https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html)

        # In[ ]:


        genresList = [
          "Action",
          "Adventure",
          "Animation",
          "Children",
          "Comedy",
          "Crime",
          "Documentary",
          "Drama",
          "Fantasy",
          "Film-Noir",
          "Horror",
          "Musical",
          "Mystery",
          "Romance",
          "Sci-Fi",
          "Thriller",
          "War",
          "Western",
          "(no genres listed)"
        ]
        def getmovie(genres):
            movieGenresList = genres.split('|')
            return movieGenresList
        def setGenresMatrix(genres):
            movieGenresMatrix = []
            movieGenresList = genres.split('|')
            for x in genresList:
                if (x in movieGenresList):
                    movieGenresMatrix.append(1)
                else:
                    movieGenresMatrix.append(0) 
            return movieGenresMatrix
        #def gettitle(genre):
        #    for i in moviesWithSimGenre.values:
         #       if genre in i['update_genres']:
          #          return i['title']
            
        movies['genresMatrix'] = movies.apply(lambda x: np.array(list(setGenresMatrix(x['genres']))), axis=1)
        movies['update_genres']= movies.apply(lambda x: np.array(list(getmovie(x['genres']))), axis=1)


        def setRatingGroup(numberOfRatings):
            # if (numberOfRatings is None): return 0
            if (1 <= numberOfRatings <= 10): return 1
            elif (11 <= numberOfRatings <= 30): return 2
            elif (31 <= numberOfRatings <= 100): return 3
            elif (101 <= numberOfRatings <= 300): return 4
            elif (301 <= numberOfRatings <= 1000): return 5
            elif (1001 <= numberOfRatings): return 6
            else: return 0

        movies['ratingGroup'] = movies.apply(lambda x: setRatingGroup(x['rating_count']), axis=1)
        movies.fillna(0, inplace=True)  # Replace NaN values to zero
        #def getmovie(genre):
            #for i in movies['genresMatrix']:
        def genreSimilarity(genre):
            moviesWithSimGenre=movies
            l=[]
            for i in moviesWithSimGenre.values:
                g=[]
                if genre in i[7]:
                    g.append(i[1])
                    g.append(i[4])
                    l.append(g)
                #print(i[7])
            #print(l)
            return l
        t=genreSimilarity(gen)
        z=slice(10)
        e=t[z]
        u=[]
        u.append(gen)
    return render_template("genre_home.html",genr=e,gen=gen)
    
if __name__=="__main__":
    app.run(debug=True)
