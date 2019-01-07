#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 10 19:29:57 2018

@author: muthuvel
"""
import re
url = "http://www.chicagotribune.com/sports/basketball/bulls/ct-spt-bulls-mailbag-mikal-bridges-michael-jordan-20180510-story.html#nt=oft01a-2la1"
def parseURL(url):
    content = []
    g = urllib.request.urlopen(url)
    soup = BeautifulSoup(g.read(), 'html.parser')
    # Article = soup.find(id='story') - denoted only the content
    
    # Classes that containg the main contents of the articles
    mydivs = soup.findAll("p")
    
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

symbol_list=['.',',','?','!', '@', '"', "'", '<', '>', '/', '[', ']','{','}','(',')',':',';', '…', '”', '#','$','%','^','&','*','-','+','_','=']

a = parseURL(url)

for b in symbol_list:           
    a = a.replace(b, '')
    a = a.replace("\n", ' ')

#index = open("../data/Unknown/Politics/metadata/index.txt")
#index.writelines(url)
#index.close()

file = open("../../data/Unknown/Sports/%d.txt" % i, "w+")
file.write(a)
file.close()


i = i + 1