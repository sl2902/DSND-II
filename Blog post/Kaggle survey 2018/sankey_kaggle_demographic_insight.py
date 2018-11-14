# prepare data for Sankey diagram
import networkx as nx
from networkx.readwrite import json_graph
import os, csv
import json

path = r'W:\Udacity\DSND-II\Blog post\Kaggle survey 2018\d3_input'
# filename = "kaggle_survey_demographic_data.csv"
# filename = "kaggle_survey_education_and_work_experience_data.csv"
# filename = "kaggle_survey_industry_and_activities_data.csv"
# filename = "kaggle_survey_industry_and_data_type.csv"
filename = "kaggle_survey_industry_and_task_type.csv"
f = open(os.path.join(path, filename), 'r')
reader = csv.reader(f, delimiter=',')

# skip header
next(reader, None)

def NodeAdd(dg, nodelist, name):
    """
    Function to prepare node list
    """
    node = len(nodelist)
    dg.add_node(node, name=name)
    nodelist.append(name)
    return dg, nodelist

dg = nx.DiGraph()
nodelist = []

for item in reader:
    for pos in range(len(item)):
        if item[pos] not in nodelist:
            dg, nodelist = NodeAdd(dg, nodelist, item[pos])

    for idx in range(len(item) - 1):
        if dg.has_edge(nodelist.index(item[idx]), nodelist.index(item[idx+1])):
            dg[nodelist.index(item[idx])][nodelist.index(item[idx+1])]['value'] += 1
        else:
            dg.add_edge(nodelist.index(item[idx]), nodelist.index(item[idx+1]), value=1)
            

json_data = json.dumps(json_graph.node_link_data(dg), indent=4, sort_keys=True) 
# out_file = 'kaggle_demographic_data.json'
# out_file = 'kaggle_education_experience_data.json'
# out_file = 'kaggle_industry_activity_data.json'
out_file = 'kaggle_industry_task_type.json'
out_path = r'W:\Udacity\DSND-II\Blog post\Kaggle survey 2018\html'
with open(os.path.join(out_path, out_file), 'w') as w:
    #json.dump(json_data, w)
    w.write(json_data)