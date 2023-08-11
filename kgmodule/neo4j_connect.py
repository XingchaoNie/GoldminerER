import os

from py2neo import Node, Relationship, NodeMatcher, Graph


class OperationNeo4j():
    def __init__(self):
        self.graph = Graph("neo4j://localhost:7687", auth=('neo4j', 'neo4j'))
        self.matcher = NodeMatcher(self.graph)

    def triplets_extraction(self):
        # TODO：更改20230716
        cypher = "CALL apoc.export.csv.query('match (n)-[r]->(m) where n:Part or n:Procedure or n:Knowledge return n, r, m', null,{stream: true,quotes:'none',useTypes:true}) YIELD data RETURN data"
       # cypher = "match (n)-[r]->(m) return n, r, m"
        raw_data = self.graph.run(cypher).data()[0]['data'][6:]
        triplets_file = os.getcwd() + '\\files/rdfs/all_data.txt'
        with open(triplets_file, 'w', encoding='utf-8') as f:
            f.write(raw_data)


    def add_part_similar(self, part1_code, part_similar_code):
        for res in part_similar_code:
            part2_code = res[0]
            similar_var = res[1]
            cypher = """match(n:Part),(m:Part) where n.PartDBID = '%s' and m.PartDBID = '%s'
                        merge(n)-[r:part_similar_to{RName:"零件相似",SimilarVar:%f}]->(m) return n,r,m""" %(part1_code, part2_code, similar_var)

            self.graph.run(cypher)


    def add_procedure_similar(self, procedure1_code, procedure_similar_code):
        for res in procedure_similar_code:
            procedure2_code = res[0]
            similar_var = res[1]
            cypher = """match(n:Procedure),(m:Procedure) where n.ProcedureDBID = '%s' and m.ProcedureDBID = '%s'
                        merge(n)-[r:procedure_similar_to{RName:"工艺相似",SimilarVar:%f}]->(m) return n,r,m""" % (procedure1_code, procedure2_code, similar_var)
            self.graph.run(cypher)



