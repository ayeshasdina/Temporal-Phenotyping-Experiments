##!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Junya Lu
@Site    :

01-Jan-2020 Dina    Modified over the time Verified Today.


"""

import pandas as pd
import numpy as np
#import networkx as nx
import os, time, random
import matplotlib.pyplot as plt
import seaborn as sns
import operator
from compiler.ast import flatten
import sys, getopt
from networkx import find_cliques,shortest_path_length,compose,spring_layout,draw_networkx,draw,weakly_connected_component_subgraphs
from networkx.classes.graph import  Graph
from networkx.classes.digraph import  DiGraph
import warnings



curr_path = os.getcwd() + '/'


def tree0(weight_value, startwindow, term):

    print 'start window:', startwindow
    # windowGraph = {}
    cliqueGraph = DiGraph()
    dic_term = {}
    dic_last_time = {}
    dic_temp = {}
    dic_term_num = {}
    dic_intersect_level = {}
    # term = 183
    
    root = 0
    cliqueGraph.add_node(root, annotation='root', windowsize='root', weight_value='root')
    w = data.shape[1]
    i = 0
    q = 0
    
    for window in range(startwindow, w):
        dic_intersect_level.clear()
        #print window ## mine
        if window == startwindow:
            

            for clique in find_cliques(windowGraph[window]):
                if len(clique) >size_clique:
                    cliqueGraph.add_node(term, annotation=list(clique), windowsize=[window],
                                         weight=weight_value)  # generate a term
                    cliqueGraph.add_edge(root, term)
                    dic_term[frozenset(clique)] = [window]  # dic_term 记录 window和clique or Dic_term records window and clique
                    dic_term_num[frozenset(clique)] = term  # dic_term_num 记录 term 序号和clique or Dic_term_num record term number and clique
                    dic_last_time[frozenset(clique)] = [window]  # dic_last_time   记录上一时刻生成的交集 用于下一时刻的比较 or Dic_last_time records the intersection generated at the last moment for comparison at the next moment
                    term = term + 1
                    print 'for start window '
                else:
                    continue
                    # print len(dic_last_time), len(dic_term), cliqueGraph.number_of_nodes()

        else:

            for clique in find_cliques(windowGraph[window]):
                if len(clique) > size_clique:
                    #print window, 'clique:', clique ## mine

                    for key, value in dic_last_time.items():  # key 是clique ,value是 [window] or Key is clique, value is [window]
                        intersect = sorted(set(key).intersection(set(clique)))
                        q = 0
                        # if len(intersect) >=  size_clique:
                        if len(intersect) >= size_clique:
                            #print 'intersect', intersect
                            # 同一层判断交集之间是否有重复的父子关系。 每生成一个交集， 判断当前层的其他term和交集的关系。or The same layer determines whether there are 
                            #duplicate parent-child relationships between intersections. Each generation of an intersection determines the relationship 
                            #between other terms and intersections of the current layer.
                            for ik, iv in dic_intersect_level.items():
                                if set(intersect) == (set(ik)):  # 生成一模一样的交集 or Generate exactly the same intersection
                                    # 判断两个的编号是否一样？or Is the two numbers the same?
                                    if dic_term_num[frozenset(key)] != dic_term_num[frozenset(ik)]:
                                        cliqueGraph.add_edge(dic_term_num[frozenset(key)], dic_term_num[frozenset(ik)])
                                    q = 1
                                    break
                                elif set(intersect).issuperset(set(ik)):  # 生成了超集 or Superset generated
                                    cliqueGraph.remove_node(dic_term_num[frozenset(ik)])
                                    dic_term.pop(frozenset(ik))  # 从四个字典中都删除该节点的信息 or Delete the node's information from all four dictionaries
                                    dic_term_num.pop(frozenset(ik))
                                    dic_intersect_level.pop(frozenset(ik))
                                    dic_temp.pop(frozenset(ik))
                                elif set(intersect).issubset(set(ik)):  # 生成了子集 or Generated subset
                                    q = 1
                                    break
                            if q == 1:
                                continue
                            dic_intersect_level[frozenset(intersect)] = 1

                            if dic_term.has_key(frozenset(intersect)):
                                # 交集已经出现过 or Intersection has appeared
                                parent = cliqueGraph.predecessors(dic_term_num[frozenset(intersect)])
                                children = cliqueGraph.successors(dic_term_num[frozenset(intersect)])
                                #print 'parent',len(parent)
                                if len(parent) > 0:
                                    # 是交集生成的term，则重定向 or  Is the intersection of generated term, then redirect
                                    cliqueGraph.add_node(term, annotation=list(intersect),
                                                         windowsize=value + [window],
                                                         weight=weight_value)
                                    for p in parent:
                                        cliqueGraph.add_edge(p, term)  # 连边 // Edge

                                    for c in children:
                                        cliqueGraph.add_edge(term, c)  # 连边 // edge
                                    cliqueGraph.remove_node(dic_term_num[frozenset(intersect)])  # 从图中删除冗余结点 or Remove redundant nodes from the figure

                                    # print 'deleted intersect nodes:',dic_term_num[frozenset(intersect)]
                                    i = i + 1
                                    dic_term.pop(frozenset(intersect))  # 字典中删除 // Delete in dictionary
                                    dic_term_num.pop(frozenset(intersect))

                                    dic_term[frozenset(intersect)] = value + [window]  # 新节点插入字典 // New node insert dictionary
                                    dic_term_num[frozenset(intersect)] = term
                                    dic_temp[frozenset(intersect)] = value + [window]  # 记录到dic_temp里 // Record to dic_temp
                                    term = term + 1
                                    continue
                                else:
                                    # 是window生成的term // Is the term generated by the window
                                    continue
                            else:
                                # 交集没有出现过， 则生成新的term // No intersection occurs, then a new term is generated
                                # print 'new term intersect never appear:', term
                                cliqueGraph.add_node(term, annotation=list(intersect), windowsize=value + [window],
                                                     weight=weight_value)  # generate a term

                                cliqueGraph.add_edge(dic_term_num[frozenset(key)], term)  # 连边，变化：只连接交集作为父亲。// Edge, change: Only connect intersections as fathers.
                                dic_term[frozenset(intersect)] = value + [window]  # 新节点插入字典 // New node insert dictionary
                                dic_term_num[frozenset(intersect)] = term
                                dic_temp[frozenset(intersect)] = value + [window]  # 记录到dic_temp里 // Record to dic_temp
                                term = term + 1
                        else:
                            continue
                else:
                    continue
            dic_last_time.clear()
            for key, value in dic_temp.items():
                dic_last_time[key] = value
            dic_temp.clear()
    print 'window', startwindow, 'size is', cliqueGraph.number_of_nodes(), cliqueGraph.number_of_edges()## mine
    # print 'deleted nodes:', i
    # fw = open('0904edges_remove.txt', 'w')
    # fw2 = open('0904terms_remove.txt', 'w')
    # fw.write('parent' + '\t' + 'child' + '\n')
    # for edge in cliqueGraph.edges():
    #     fw.write(str(edge[0]) + '\t' + str(edge[1]) + '\n')
    # fw.close()
    # fw2.write('term_id' + '\t' + 'anno_genes' + '\t' + 'window' + '\t' + 'gene_size' + '\t' + 'window_size' + '\n')
    # for key, value in dic_term.items():
    #     fw2.write(str(dic_term_num[key]) + '\t' + str(key) + '\t' + str(value) + '\t' + str(len(key)) + '\t' + str(len(value)) + '\n')
    # fw2.close()
    # for nodes in cliqueGraph.nodes():
    #     if cliqueGraph.degree(nodes) == 0:
    #         print nodes
 
    return cliqueGraph, dic_term, dic_term_num, term


def sign_value(node, gene_set, window):

    # window_set = [i for i in range(min(window) + 49, max(window) + 49 + 10)]  # 回到最原始的数据上
    window_set = window
    #print phe1
    p1 = phe1.loc[gene_set, window_set]
    p2 = phe2.loc[gene_set, window_set]
    p3 = phe3.loc[gene_set, window_set]

    p1_list = flatten(p1.values.tolist())
    p2_list = flatten(p2.values.tolist())
    p3_list = flatten(p3.values.tolist())
    sign_p1 = [x for x in p1_list if x < phe1_t1 or x > phe1_t2]
    sign_p2 = [x for x in p2_list if x < phe2_t1 or x > phe2_t2]
    sign_p3 = [x for x in p3_list if x < phe3_t1 or x > phe3_t2]
    sign = len(sign_p1) + len(sign_p2) + len(sign_p3)
    all = len(p1_list) + len(p2_list) + len(p3_list)
    purity = float(sign) / all
    # purity1 = len(sign_p1) / float(len(p1_list))
    # purity2 = len(sign_p2) / float(len(p2_list))
    # purity3 = len(sign_p3) / float(len(p3_list))
    # if purity1 >= pp or purity2 >= pp or purity3 >= pp:
    #     purity = max(purity1, purity2, purity3)
    return sign, purity
def sign_value0(node, gene_set, window):
    t_min = min(window) + first_time_point 
    t_max = max(window) + first_time_point + window_size 
    window_set = [i for i in range(t_min, t_max)]  # 回到最原始的数据上 // Back to the most original data
    p1 = phe1.loc[gene_set, window_set]
    p2 = phe2.loc[gene_set, window_set]
    p3 = phe3.loc[gene_set, window_set]
    p1_list = flatten(p1.values.tolist())
    p2_list = flatten(p2.values.tolist())
    p3_list = flatten(p3.values.tolist())
    # sign_p1 = [x for x in p1_list if x <= b1 or x >= u1]
    # sign_p2 = [x for x in p2_list if x <= b2 or x >= u2]
    # sign_p3 = [x for x in p3_list if x <= b3 or x >= u3]
    sign_p1 = [x for x in p1_list if x < phe1_t1 or x > phe1_t2]
    sign_p2 = [x for x in p2_list if x < phe2_t1 or x > phe2_t2]
    sign_p3 = [x for x in p3_list if x < phe3_t1 or x > phe3_t2]

    sign = len(sign_p1) + len(sign_p2) + len(sign_p3)
    all = len(p1_list) + len(p2_list) + len(p3_list)
    if p1[t_min].mean() - p1[t_max - 1].mean() < 0:
        trend1 = 1
    else:
        trend1 = 0
    if p2[t_min].mean() - p2[t_max - 1].mean() < 0:
        trend2 = 1
    else:
        trend2 = 0
    if p3[t_min].mean() - p3[t_max - 1].mean() < 0:
        trend3 = 1
    else:
        trend3 = 0
    purity = float(sign) / all
    purity1 = len(sign_p1) / float(len(p1_list))
    purity2 = len(sign_p2) / float(len(p2_list))
    purity3 = len(sign_p3) / float(len(p3_list))
    # if purity1 >= pp or purity2 >= pp or purity3 >= pp:
    #     purity = max(purity1, purity2, purity3)

    return sign, purity

if __name__ == '__main__':

    start = time.clock()
   

    inputfile = 'input/weight_win_7.txt'
    phe1_file = 'input/common_name_phi2.txt'
    phe2_file = 'input/common_name_qet.txt'
    phe3_file = 'input/common_name_qit.txt'
    
    weight_value = 0.7 #NPM cluster membership overlap rate threshold
    window_size =7    # min number of time points of an event
    size_clique = 5   #Min number of plants of an event
    sign_score = 0.5   #threshold of the Ratio of significant phebnotype values in an event
    
    phe1_t1 = -0.20  #....phi2 lower bound
    phe1_t2 = 0.12  # phi2 upper bound
    
    phe2_t1 =-0.15 # qet lower bound
    phe2_t2 = 0.15 # qet upper bound
    
    phe3_t1 =-0.42   # qit lower bound 
    phe3_t2 = 0.35 # qit upper bound
  
    

    first_time_point = 1  #the starting point where events can be generated
            
    term = 329  # number of plants
            

        

    # python project2.py -i Graph1.txt -w 0.8 -s 1 -p 0.5 -m 0.5 -n 0.7 -b 0.2 -v 0.7 -c 0.1 -z 1
    print phe1_t1, phe2_t1, phe3_t1
    print size_clique
    print weight_value
    print '---begin---'
    ##### what does mean??
    # filename = 'Graph1.txt'
    data = pd.read_csv(inputfile, index_col=0, sep='\t')
    columns_num = int(data.shape[1])

    phe1 = pd.read_table(phe1_file,index_col=0,sep=",")
    phe2 = pd.read_table(phe2_file,index_col=0,sep=",")
    phe3 = pd.read_table(phe3_file,index_col=0,sep=",")
    phe1.columns = [i for i in range(0, phe1.shape[1])]
    phe2.columns = [i for i in range(0, phe2.shape[1])]
    phe3.columns = [i for i in range(0, phe3.shape[1])]

    curr_path = os.getcwd() + '/'
    windowGraph = {}
    # output undirected graph
    # for i in range(0, columns_num):
    #     tree_statis(weight_value, i)

    s = 0

    for window in range(0, columns_num):
        windowGraph[window] = Graph()
        df = data[data[data.columns[window]] >= (weight_value + 0.00001)]

        # print window, weight_value, df.shape
        for edge in range(0, df.shape[0]):
            node_1, node_2 = df.index[edge].split('-')
            windowGraph[window].add_edge(node_1, node_2)
    # 先产生第一个window的 tree // First generate the first window of the tree
    # print windowGraph[0].number_of_nodes(), windowGraph[0].number_of_edges()
    cliqueGraph0, dic_term0, dic_term_num0, term = tree0(weight_value, 0, term)
    dic_all = {}
    dic_all = dic_term0.copy()
    dic_all_term_num = dic_term_num0.copy()
    copy_clique = cliqueGraph0
    for i in range(1, columns_num):
        print 'begin term num:', term
        cliqueGraph1, dic_term1, dic_term_num1, term = tree0(weight_value, i, term)
        cliqueGraph0 = compose(cliqueGraph0, cliqueGraph1)
        # 判断冗余信息 // Judging redundant information
        for key in dic_term1.keys(): # dic_term1 当前时间生成的结点信息 geneList:windowList // Dic_term1 Node information generated at the current time geneList:windowList
            if dic_all.has_key(key): # dic_all 之前时间点生成的结点信息 累计 // Node information generated before time dic_all Accumulated
                # print 'exit key',key
                # 如果之前 已经生成 过这样的geneList // If you have previously generated such a geneList
                if set(dic_all[key]).issuperset(set(dic_term1[key])):
                    # 如果之前的时间片段较长 就舍弃当前这个term // If the previous time segment is long, discard the current term
                    dic_term1.pop(key)
                    num = dic_term_num1[key]
                    cliqueGraph0.remove_node(num)
                    dic_term_num1.pop(key)
                    s = s + 1
                    # print 'remove node', num
                else:
                    #如果现在的时间片段长或者是两者没有子集关系 则 合并时间片段 并舍弃当前这个term // If the current time segment is long or there is no subset relationship between the two, merge the time segment and discard the current term
                    dic_all[key] = dic_all[key] + dic_term1[key]
                    dic_term1.pop(key)
                    num = dic_term_num1[key]
                    cliqueGraph0.remove_node(num)
                    dic_term_num1.pop(key)
                    # print 'update node',dic_all_term_num[key], 'and remove node', num

            else:
                # flag 连最上层子结点 子节点的孩子不需要再连接 // Flag does not need to connect with the children of the top child child nodes

                dic_this_edge = {}
                for old in dic_all.keys():  # 对之前已经存在的结点 // For previously existing nodes
                    old_id = dic_all_term_num[old]
                    this_id = dic_term_num1[key]
                    flag1 = 0
                    flag2 = 0
                    if set(key).issuperset(old) and set(dic_all[old]).issuperset(dic_term1[key]):
                        # 如果当前结点的注释基因是它的父集 注释时间是它的子集 // If the current node's annotation gene is its parent, the annotation time is a subset of it.
                        # 判断old的parent是否和 key已经连接 // Determine if the parent of the old is connected to the key
                        try:
                            parent = cliqueGraph0.predecessors(old_id)
                        except:
                            continue

                        dic_parent = {}
                        for p in parent:
                            dic_parent[p] = len(cliqueGraph0.node[p]['annotation'])
                        dic2 = sorted(dic_parent.items(), key=operator.itemgetter(1), reverse=True)
                        parent = []
                        for pair in dic2:
                            parent.append(pair[0])
                        for p_id in parent:
                            try:
                                p_anno = cliqueGraph0.node[p_id]['annotation']
                                p_wind = cliqueGraph0.node[p_id]['windowsize']

                                if set(key).issuperset(set(p_anno)) and set(p_wind).issuperset(dic_term1[key]):
                                    flag1 = 1
                                    break
                                    # print p_id, old, this_id
                                if dic_this_edge[(this_id, p_id)] == 1:
                                    flag1 = 1
                                    break
                            except:
                                continue
                        if flag1 == 0:
                            # key--> old
                            if this_id!=old_id and not cliqueGraph0.has_edge(this_id, old_id):
                                cliqueGraph0.add_edge(this_id, old_id)
                                dic_this_edge[(this_id, old_id)] = 1
                                break
                            # print 'add edge', this_id, old_id
                        else:
                            # 已经连接到它的父亲上了 // Already connected to its father
                            continue

                    elif set(key).issuperset(old) and  set(dic_term1[key]).issuperset(dic_all[old]):
                        # 如果当前结点的注释基因是它的父集 注释时间也是它的父集  delete old // If the current node's annotation gene is its parent, the annotation time is also its parent set. delete old

                        child = cliqueGraph0.successors(old_id)
                        for c in child:
                            if not cliqueGraph0.has_edge(this_id, c) and len(cliqueGraph0.node[c]['windowsize']) > len(
                                    dic_term1[key]):
                                cliqueGraph0.add_edge(this_id, c)

                        cliqueGraph0.remove_node(old_id)
                        dic_all.pop(old)
                        dic_all_term_num.pop(old)
                        # print 'remove node', old_id
                        ###
                    elif set(old).issuperset(key) and set(dic_all[old]).issuperset(dic_term1[key]):
                        # 如果当前结点的注释基因是它的子集 注释时间也是它的子集 delete this // If the current node's annotation gene is a subset of it, the annotation time is also a subset of it. delete this
                        child = cliqueGraph0.successors(this_id)
                        for c in child:
                            if not cliqueGraph0.has_edge(old_id, c) and len(cliqueGraph0.node[c]['windowsize']) > len(
                                    dic_all[old]):
                                cliqueGraph0.add_edge(old_id, c)

                        cliqueGraph0.remove_node(this_id)
                        dic_term_num1.pop(this_id)
                        dic_term1.pop(key)
        dic_all.update(dic_term1)
        dic_all_term_num.update(dic_term_num1)
    print 'raw size--------', cliqueGraph0.number_of_nodes(), cliqueGraph0.number_of_edges()

    print 'start calculate purity...'
    dic_term_score = {}
    delete_leave_flag1 = 0
    delete_sum = 0
    for node in cliqueGraph0.nodes():
        if node == 0:
            continue
        else:
            gene_set = cliqueGraph0.node[node]['annotation']
            window_set = cliqueGraph0.node[node]['windowsize']
            # 判断phenotype是否有意义 // Determine if phenotype makes sense
            sign, score = sign_value0(node, gene_set, window_set)

            if score < sign_score:
                # 无意义，delete，重定向 //Meaningless, delete, redirect
                parent = cliqueGraph0.predecessors(node)
                child = cliqueGraph0.successors(node)
                if len(child) == 0:
                    delete_leave_flag1 += 1
                for p in parent:
                    for c in child:
                        if not cliqueGraph0.has_edge(p, c):
                            cliqueGraph0.add_edge(p, c)
                cliqueGraph0.remove_node(node)
                delete_sum += 1
            else:
                dic_term_score[node] = score
                continue
    # print 'after purity window', cliqueGraph0.number_of_nodes(), cliqueGraph0.number_of_edges()
    for node in cliqueGraph0.nodes():
        if node == 0:
            continue
        else:
            parent = cliqueGraph0.predecessors(node)
            if len(parent) > 1 and 0 in parent:
                cliqueGraph0.remove_edge(0, node)

    # print 'before remove redundant ', cliqueGraph0.number_of_nodes(), cliqueGraph0.number_of_edges()
    # redudant
    print 'start remove redundant nodes between same level...'
    OntologyGraph = cliqueGraph0
    distance = shortest_path_length(cliqueGraph0, source=0)
    df = pd.DataFrame(distance.items())
    grouped = df.groupby(1, sort=False)
    dic_level = {}
    # 将nodes按照level分层 // Layer nodes by level
    for g in grouped:
        l = g[0]
        nodes = g[1][0].tolist()
        dic_level[l] = nodes
    for level in range(1, l):
        for i in range(0, len(dic_level[level])):
            for j in range(i + 1, len(dic_level[level])):
                try:
                    term1 = dic_level[level][i]
                    term2 = dic_level[level][j]
                    gene1 = cliqueGraph0.node[term1]['annotation']
                    gene2 = cliqueGraph0.node[term2]['annotation']
                    time1 = sorted(cliqueGraph0.node[term2]['windowsize'])
                    time2 = sorted(cliqueGraph0.node[term2]['windowsize'])
                except:
                    continue

                if len(set(gene2) | (set(gene1))) - len(set(gene2) & (set(gene1))) <= 2 and time1 == time2:
                    # print term2, gene2, time2
                    # print term1, gene1, time1

                    try:
                        # parent2 = cliqueGraph0.predecessors(term2)
                        child2 = cliqueGraph0.successors(term2)

                        cliqueGraph0.node[term1]['geneSet'] = list(set(gene2) | (set(gene1)))
                        # loop
                        # for p in parent2:
                        #     if p != term1:
                        #         cliqueGraph0.add_edge(p, term1)
                        for c in child2:
                            if c != term1 and not cliqueGraph0.has_edge(c, term1):
                                cliqueGraph0.add_edge(term1, c)
                        cliqueGraph0.remove_node(term2)
                        # print term2
                        sum = sum + 1
                    except:
                        continue

    dic_term_score = {}
    for node in cliqueGraph0.nodes():
        if node == 0:
            continue
        else:
            gene_set = cliqueGraph0.node[node]['annotation']
            window_set = cliqueGraph0.node[node]['windowsize']
            # 判断phenotype是否有意义 // Determine if phenotype makes sense

            sign, score = sign_value0(node, gene_set, window_set)

            if score < sign_score:
                # 无意义，delete，重定向 // Meaningless, delete, redirect
                parent = cliqueGraph0.predecessors(node)
                child = cliqueGraph0.successors(node)
                for p in parent:
                    for c in child:
                        if not cliqueGraph0.has_edge(p, c):
                            cliqueGraph0.add_edge(p, c)
                cliqueGraph0.remove_node(node)
                # print 'purity is low...then remove this node:', node
            else:
                dic_term_score[node] = score
                continue

    for node in cliqueGraph0.nodes():
        if node == 0:
            continue
        else:
            parent = cliqueGraph0.predecessors(node)
            if len(parent) > 1 and 0 in parent:
                cliqueGraph0.remove_edge(0, node)
    loop_edges = cliqueGraph0.selfloop_edges()
    cliqueGraph0.remove_edges_from(loop_edges)
    cliqueGraph0 = max(weakly_connected_component_subgraphs(cliqueGraph0), key=len)

    fw1 = open('phenotype_ontology.obo', 'w')
    for node in cliqueGraph0.nodes():
        if node != 0:
            fw1.write('[Term]' + '\n')
            fw1.write('id:' + 'TPO:' + str(node) + '\n')
            fw1.write('purity:' + str(round(dic_term_score[node], 4)) + '\n')
            fw1.write('number of annotation genes:' + str(len(cliqueGraph0.node[node]['annotation'])) + '\n')
            fw1.write('number of annotation windows:' + str(len(cliqueGraph0.node[node]['windowsize'])) + '\n')
            parent = set(cliqueGraph0.predecessors(node))
            for p in parent:
                fw1.write('is_a:' + 'TPO:' + str(p) + '\n')
            fw1.write('\n')
        else:
            fw1.write('[Term]' + '\n' + 'id:TPO:0' + '\n' + '\n')

    fw2 = open('annotation_win_7.txt', 'w')
    fw2.write(
        'term_id' + '\t' + 'sign_score' + '\t' + 'level' + '\t' + 'annotation_gene' + '\t' + 'start_time' + '\t'
        + 'end_time' + '\t' + 'geneSize' + '\t' + 'windowSize' + '\n')

    for node in cliqueGraph0.nodes():
        if node == 0:
            continue
        else:
            try:
                fw2.write(
                    'TPO:' + str(node) + '\t' +
                    str(round(dic_term_score[node], 4)) + '\t' +
                    str(shortest_path_length(cliqueGraph0, 0, node)) + '\t' +
                    ','.join(cliqueGraph0.node[node]['annotation']) + '\t' +
                    str(min(cliqueGraph0.node[node]['windowsize'])+first_time_point) + '\t' +
                    str(max(cliqueGraph0.node[node]['windowsize'])+first_time_point + window_size) + '\t' +
                    str(len(cliqueGraph0.node[node]['annotation'])) + '\t' +
                    str(len(cliqueGraph0.node[node]['windowsize'])) + '\n')
            except:
                cliqueGraph0.add_edge(0, node)
                fw2.write(
                    'TPO:' + str(node) + '\t' +
                    str(round(dic_term_score[node], 4)) + '\t' +
                    str(shortest_path_length(cliqueGraph0, 0, node)) + '\t' +
                    ','.join(cliqueGraph0.node[node]['annotation']) + '\t' +
                    str(min(cliqueGraph0.node[node]['windowsize'])+first_time_point) + '\t' + 
                    str(max(cliqueGraph0.node[node]['windowsize'])+first_time_point + window_size) + '\t' +
                    str(len(cliqueGraph0.node[node]['annotation'])) + '\t' +
                    str(len(cliqueGraph0.node[node]['windowsize'])) + '\n')
    end = time.clock()
    fw1.close()
    fw2.close()
    print 'FINALLY----------', cliqueGraph0.number_of_nodes(), cliqueGraph0.number_of_edges()
    # draw picture
    if cliqueGraph0.number_of_nodes() < 20: ### Number of Maximum Node.
        pos = spring_layout(cliqueGraph0)
        draw_networkx(cliqueGraph0, pos, with_labels=True, node_size=600)
        for node in cliqueGraph0.nodes():
            if node!=0:
                cliqueGraph0.node[node]['windowsize'] = [i+1 for i in cliqueGraph0.node[node]['windowsize']]
            x, y = pos[node]
            plt.text(x, y + 0.1, s=cliqueGraph0.node[node]['annotation'],
                        bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')
            plt.text(x, y + 0.15, s=cliqueGraph0.node[node]['windowsize'],
                        bbox=dict(facecolor='red', alpha=0.5), horizontalalignment='center')
        plt.title('Phenotype Ontology', fontsize=20)
        plt.xlabel('( This figure is generated when the number of nodes in PO smaller than 20 )', fontsize=15)
        plt.show()
    print 'The function run time is : %.03f seconds' % (end - start)
