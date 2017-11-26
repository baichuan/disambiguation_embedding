import networkx as nx


class DataSet():

    def __init__(self, file_path):

        self.file_path = file_path
        self.paper_authorlist_dict = {}
        self.paper_list = []
        self.coauthor_list = []
        self.label_list = []
        self.C_Graph = nx.Graph()
        self.D_Graph = nx.Graph()
        self.num_nnz = 0

    def reader_arnetminer(self):
        paper_index = 0
        coauthor_set = set()

        with open(self.file_path, "r") as filetoread:
            for line in filetoread:
                line = line.strip()
                if "FullName" in line:
                    ego_name = line[line.find('>')+1:line.rfind('<')].strip()
                elif "<publication>" in line:
                    paper_index += 1
                    self.paper_list.append(paper_index)
                elif "<authors>" in line:
                    author_list = line[line.find('>')+1: line.rfind('<')].strip().split(',')
                    if len(author_list) > 1:
                        if ego_name in author_list:
                            author_list.remove(ego_name)
                            self.paper_authorlist_dict[paper_index] = author_list
                        else:
                            self.paper_authorlist_dict[paper_index] = author_list

                        for co_author in author_list:
                            coauthor_set.add(co_author)

                        # construct the coauthorship graph
                        for pos in xrange(0, len(author_list) - 1):
                            for inpos in xrange(pos+1, len(author_list)):
                                src_node = author_list[pos]
                                dest_node = author_list[inpos]
                                if not self.C_Graph.has_edge(src_node, dest_node):
                                    self.C_Graph.add_edge(src_node, dest_node, weight = 1)
                                else:
                                    edge_weight = self.C_Graph[src_node][dest_node]['weight']
                                    edge_weight += 1
                                    self.C_Graph[src_node][dest_node]['weight'] = edge_weight
                    else:
                        self.paper_authorlist_dict[paper_index] = []
                elif "<label>" in line:
                    label = int(line[line.find('>')+1: line.rfind('<')].strip())
                    self.label_list.append(label)

        self.coauthor_list = list(coauthor_set)
        """
        compute the 2-extension coauthorship for each paper
        generate doc-doc network
        edge weight is based on 2-coauthorship relation
        edge weight details are in paper definition 3.3
        """
        paper_2hop_dict = {}
        for paper_idx in self.paper_list:
            temp = set()
            if self.paper_authorlist_dict[paper_idx] != []:
                for first_hop in self.paper_authorlist_dict[paper_idx]:
                    temp.add(first_hop)
                    if self.C_Graph.has_node(first_hop):
                        for snd_hop in self.C_Graph.neighbors(first_hop):
                            temp.add(snd_hop)
            paper_2hop_dict[paper_idx] = temp

        for idx1 in xrange(0, len(self.paper_list) - 1):
            for idx2 in xrange(idx1 + 1, len(self.paper_list)):
                temp_set1 = paper_2hop_dict[self.paper_list[idx1]]
                temp_set2 = paper_2hop_dict[self.paper_list[idx2]]

                edge_weight = len(temp_set1.intersection(temp_set2))
                if edge_weight != 0:
                    self.D_Graph.add_edge(self.paper_list[idx1],
                                          self.paper_list[idx2],
                                          weight = edge_weight)
        bipartite_num_edge = 0
        for key, val in self.paper_authorlist_dict.items():
            if val != []:
                bipartite_num_edge += len(val)

        self.num_nnz = self.D_Graph.number_of_edges() + \
                       self.C_Graph.number_of_edges() + \
                       bipartite_num_edge


    def reader_citeseerx(self):
        paper_index = 0
        coauthor_set = set()

        with open(self.file_path, "r") as filetoread:
            for line in filetoread:
                linetuple = line.strip().split(',')
                if paper_index == 0:
                    ego_name = linetuple[1] 

                paper_index += 1
                self.paper_list.append(paper_index)
                
                author_list = linetuple[1:]
                if len(author_list) > 1:
                    if ego_name in author_list:
                        author_list.remove(ego_name)
                        self.paper_authorlist_dict[paper_index] = author_list
                    else:
                        self.paper_authorlist_dict[paper_index] = author_list

                    for co_author in author_list:
                        coauthor_set.add(co_author)

                    # construct the coauthorship graph
                    for pos in xrange(0, len(author_list) - 1):
                        for inpos in xrange(pos+1, len(author_list)):
                            src_node = author_list[pos]
                            dest_node = author_list[inpos]
                            if not self.C_Graph.has_edge(src_node, dest_node):
                                self.C_Graph.add_edge(src_node, dest_node, weight = 1)
                            else:
                                edge_weight = self.C_Graph[src_node][dest_node]['weight']
                                edge_weight += 1
                                self.C_Graph[src_node][dest_node]['weight'] = edge_weight
                else:
                    self.paper_authorlist_dict[paper_index] = []
                        
                label = int(linetuple[0])
                self.label_list.append(label)

        self.coauthor_list = list(coauthor_set)
        """
        compute the 2-extension coauthorship for each paper
        generate doc-doc network
        edge weight is based on 2-coauthorship relation
        edge weight details are in paper definition 3.3
        """
        paper_2hop_dict = {}
        for paper_idx in self.paper_list:
            temp = set()
            if self.paper_authorlist_dict[paper_idx] != []:
                for first_hop in self.paper_authorlist_dict[paper_idx]:
                    temp.add(first_hop)
                    if self.C_Graph.has_node(first_hop):
                        for snd_hop in self.C_Graph.neighbors(first_hop):
                            temp.add(snd_hop)
            paper_2hop_dict[paper_idx] = temp

        for idx1 in xrange(0, len(self.paper_list) - 1):
            for idx2 in xrange(idx1 + 1, len(self.paper_list)):
                temp_set1 = paper_2hop_dict[self.paper_list[idx1]]
                temp_set2 = paper_2hop_dict[self.paper_list[idx2]]

                edge_weight = len(temp_set1.intersection(temp_set2))
                if edge_weight != 0:
                    self.D_Graph.add_edge(self.paper_list[idx1],
                                          self.paper_list[idx2],
                                          weight = edge_weight)
        bipartite_num_edge = 0
        for key, val in self.paper_authorlist_dict.items():
            if val != []:
                bipartite_num_edge += len(val)

        self.num_nnz = self.D_Graph.number_of_edges() + \
                       self.C_Graph.number_of_edges() + \
                       bipartite_num_edge