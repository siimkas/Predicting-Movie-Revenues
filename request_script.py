import requests
import csv
import cpi
import pandas as pd

cpi.update()


api_key = 'your api key here'
search_payload = {'api_key': api_key, 'sort_by':'revenue.desc','include_adult':False,'include_video':False,
'primary_release_date.gte':'2010-01-01', 'primary_release_date.lte':'2019-12-31','vote_count.gte':100 }
r = requests.get('https://api.themoviedb.org/3/discover/movie',params=search_payload, timeout=10).json()

total_pages=r['total_pages']
total_results=r['total_results']

data_file = open('movies_data.csv', 'w',encoding="utf-8",newline='')

csv_writer = csv.writer(data_file)

keys = ['title', 'budget','original_language','popularity','release_date','runtime','vote_average','revenue']
csv_writer.writerow(keys)

for page in range(1,total_pages+1):
    search_payload['page']=page
    print(search_payload['page'])
    r = requests.get('https://api.themoviedb.org/3/discover/movie',params=search_payload, timeout=10).json()
    for movie in r['results']:

        
            movie_id = movie['id']
            movie_data = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}', headers={'Accept-Encoding': 'identity'}  ).json()  
            
            if(movie_data['budget']>0 and movie_data['revenue']>0):
                data = []
                for key in keys:
                    data.append(movie_data[key])
                csv_writer.writerow(data)
        
data_file.close()



search_payload = {'api_key': api_key, 'sort_by':'revenue.desc','include_adult':False,'include_video':False,
'primary_release_date.gte':'2000-01-01', 'primary_release_date.lte':'2009-12-31','vote_count.gte':100 }

data_file = open('movies_data.csv', 'a',encoding="utf-8",newline='')

r = requests.get('https://api.themoviedb.org/3/discover/movie',params=search_payload, timeout=10).json()

total_pages=r['total_pages']
total_results=r['total_results']

csv_writer = csv.writer(data_file)

for page in range(1,total_pages+1):
    search_payload['page']=page
    print(search_payload['page'])
    r = requests.get('https://api.themoviedb.org/3/discover/movie',params=search_payload, timeout=10).json()
    for movie in r['results']:

        
            movie_id = movie['id']
            movie_data = requests.get(f'https://api.themoviedb.org/3/movie/{movie_id}?api_key={api_key}',headers={'Accept-Encoding': 'identity'}).json()  
            
            if(movie_data['budget']>0 and movie_data['revenue']>0):
                data = []
                for key in keys:
                    data.append(movie_data[key])
                csv_writer.writerow(data)
        
data_file.close()

df = pd.read_csv('movies_data.csv')
def adjust_to_inflation(data, column):
    
    return data.apply(lambda x: cpi.inflate(x[column], int(x.release_date[0:4])), axis=1)
df['adjusted_budget'] = adjust_to_inflation(df, 'budget')
df['adjusted_revenue'] = adjust_to_inflation(df, 'revenue')

df.to_csv('movies_data.csv')