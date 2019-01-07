#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 17:24:54 2018

@author: muthuvel
"""
from nytimesarticle import articleAPI
from bs4 import BeautifulSoup
import urllib
import time

# Define API key and input parameters for extraction
api = articleAPI('fa567ce571174336957fc6786b4dc91e')
category = "Health"
search_keywords = ["Doctor","Medical","Patient","flu","disease","Medicine","Pharamaceutical","Healthcare","Insurance"]
keyword = "/"+category+"/"
keyword1 = "Testing/"+category
Pages = [16,17,18,19,20,21,22,23,24,25,26,27,28,29,30] #define how many pages you want to extract
Date = 20110202
# doctor dental hygience


# Method to extract the content of the url
def parseURL(url):
    content = []
    g = urllib.request.urlopen(url)
    soup = BeautifulSoup(g.read(), 'html.parser')
    # Article = soup.find(id='story') - denoted only the content
    
    # Classes that containg the main contents of the articles
    mydivs = soup.findAll("p", {"class": "css-1cy1v93 e2kc3sl0"})
    
    # For articles in which  the above class extraction command fails
    if (mydivs == []):
        mydivs = soup.findAll("p", {"class": "story-body-text story-content"})
    
    if (mydivs != []):
        # Adding title to the content
        content = soup.title.text
        #return []
        
    for j in range(0,len(mydivs)):
        content = content + '\n' + mydivs[j].text 
    
    return content
        

# Main code begins
def collectArticles(PAGE, DATE, search_keyword, keyword, category):
    print('Collecting articles from page:%d' % PAGE)
    articles = api.search(q=search_keyword, begin_date = DATE, page=PAGE)
    response = articles['response']
    docs = response['docs']
    
    # Index contains the metadata - url of all the articles collected so far
    index = open("../../data/%s/metadata/index.txt" %(category),"r")
    
    # Creating an index file if this the first time articles are collected on a topic
    if (index.readlines() == []):
        index = open("../../data/%s/metadata/index.txt" %(category),"w+")
        web_url=[]
        for i in range(0,len(docs)):
            if (keyword.lower() in docs[i]['web_url']): #Checks if articles in from the relevant category
                web_url.append(docs[i]['web_url'])
                index.writelines("%s\n" % docs[i]['web_url'])
    index.close()  
    
    # Reading index file
    index = open("../../data/%s/metadata/index.txt" %(category),"r")
    web_url = index.read()
    web_url = web_url.splitlines()
    
    # Appending all collected articles to the existing URLs and saving to the index file
    for i in range(0,len(docs)):
        if (keyword.lower() in docs[i]['web_url']): #Checks if articles in from the relevant category
            web_url.append(docs[i]['web_url'])
    web_url = list(set(web_url))    #removes duplicates
    index = open("../../data/%s/metadata/index.txt" %(category),"w+")
    for i in range(0,len(web_url)):
            index.writelines("%s\n" % web_url[i])
    index.close()  
    print("Articles successfully collected from page:%d and appended to index file" % PAGE)
    
    return web_url
    
def main(Pages, Date, search_keywords, keyword, category):
    print("###################################")
    for idx in range(0,len(search_keywords)):
        search_keyword = search_keywords[idx]
        for page in Pages:
            web_url = collectArticles(page, Date, search_keyword, keyword, category)
            time.sleep(5)
    
    # Extract contents of url one by one and write it to text file
    print("###################################")
    print("Scraping all the urls")
    j=0
    i=0
    attempt = 0 # Used to reparse articles that could not be parsed due to API issues
    while (i<len(web_url)):
        #print(i)
        try:
            Article_content = parseURL(web_url[i])
            if (Article_content == []):
                attempt += 1
                #print("Article %d extraction failed" %i + " - trying again")
                if (attempt == 8):
                    print("Server busy - Article %d extraction failed" %i)
                    attempt = 0
                    i=i+1
                continue
            f = open("../../data/%s/%s.txt" %(category, j),"w+")
            f.write(Article_content)
            f.close()
        except:
            #web_url.remove(web_url[i])
            print("Article %d" %i + " skipped - unable to fetch data")
            #i = i-1
            j = j-1
        j= j+1
        i= i+1
        
    print("###################################")
    print("Total number of articles extracted:%d" % j)
    #index.write(str())
    #index.close()
    
if __name__ == "__main__":
    print("File loaded directly")
    main(Pages, Date, search_keywords, keyword, keyword1)