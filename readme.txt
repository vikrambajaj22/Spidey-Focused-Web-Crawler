CS-GY 6913: Web Search Engines - Assignment 1 - Focused Crawler
===============================================================
Vikram Sunil Bajaj (vsb259)

Files:
------
spidey.py: Python code
spidey_multiThread.py: An attempt with multi-threading (results not shown for this code)
crawler_log_1.txt: Log file for a BFS crawl for the query 'wildfires california'
crawler_log_2.txt: Log file for a Focused crawl for the query 'wildfires california'
crawler_log_3.txt: Log file for a BFS crawl for the query 'brooklyn dodgers'
crawler_log_4.txt: Log file for a Focused crawl for the query 'brooklyn dodgers'
readme.txt: This file
explain.txt: Documentation for the important Classes and functions in spidey.py, along with assumptions, known bugs/missing features and extra feautres

Program Input:
--------------
The program will ask for the following inputs:
1. The query: The search string (default: 'wildfires california')
2. The number of start pages: The number of start pages to be retrieved from the initial Google search (default: 10)
3. The number of pages to be returned: The max. number of pages to be crawled (minimum: 10, default: 1000)
4. The max. number of links to be fetched from each page: The links-per-page limit (default: 25)
5. The mode: The crawl mode i.e. 'bfs' (default) or 'focused'
6. The relevalce threshold: A threshold between 0 and 4.75 (default: 1) [default is 1 because for 1000 links, most of them after the first 25 or so will have a relevance < 3]

Expected Output:
----------------
crawler_log file with crawl parameters, details about the crawled links, harvest rate, time elapsed and errors encountered 