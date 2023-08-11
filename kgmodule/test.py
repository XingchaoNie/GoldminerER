import pymysql

class OperationMysql():
    def __init__(self):
        # MySQL
        self.conn = pymysql.connect(
            host="127.0.0.1",
            port=3306,
            user="root",
            passwd="123456",
            db="MT",
            # charset="utf-8",
            cursorclass=pymysql.cursors.DictCursor
        )
        # DM
        # self.conn = dmPython.connect(
        #     server="127.0.0.1",
        #     port=5236,
        #     user="SYSDBA",
        #     password="123456789",
        #     # db="MT",
        #     # charset="utf-8",
        #     # cursorclass=pymysql.cursors.DictCursor
        # )
        self.cur = self.conn.cursor()

    def search_all(self, sql):
        self.cur.execute(sql)
        result = self.cur.fetchall()
        return result

def generate_sql(data_list):
    data = data_list[0]
    cols = ", ".join('`{}`'.format(k) for k in data.keys())
    val_cols = ', '.join('%({})s'.format(k) for k in data.keys())
    sql = """
    INSERT INTO student(%s) VALUES(%s)
    """ % (cols, val_cols)
    return sql

def test():
    student_list = [
        {'name': 'Tony', 'age': 19, 'sex': 'male'},
        {'name': 'Lisa', 'age': 18, 'sex': 'female'},
        {'name': 'Jack', 'age': 20, 'sex': 'male'}
    ]
    op = OperationMysql()
    sql = generate_sql(student_list)
    op.cur.executemany(sql, student_list)
    op.conn.commit()
    op.cur.close()
    op.conn.close()

if __name__ == '__main__':
    op = OperationMysql()
    d_1 = {'h': '导轨', 't': '机床', 'r': '因果关系'}
    d_2 = {'h_type': 'knowledge', 't_type': 'knowledge', 'r_type': '因果关系'}
    # sql = 'INSERT into t_temp(SysNo, RDFS, TYPES) VALUES(%s, %s, %s)' % (str(200), str(d_1), str(d_2))
    sql = """
    INSERT into t_temp(SysNo, RDFS, TYPES) VALUES(200, "{'h': '导轨', 't': '机床', 'r': '因果关系'}", "{'h_type': 'knowledge', 't_type': 'knowledge', 'r_type': '因果关系'}")
    """
    # sql = """
    #     INSERT into t_temp(SysNo, RDFS, TYPES) VALUES("200",
    # """ + str(d_1) + """
    #     ,
    # """ + str(d_2) + """
    #     )
    # """
    op.cur.execute(sql)
    op.conn.commit()
    # print(str(d_1))
    # test()
    sql = 'INSERT into t_temp(SysNo, RDFS, TYPES) VALUES(%s, %s, %s)' % (str(200), "\"" + str(d_1) + "\"", "\"" + str(d_2) + "\"")
    # sql = 'dasdas'.join("ppppppppppa")
    print(sql)

    {'h_node': '导轨', 't_node': '机床', 'relation': '因果关系'}
    {'h_type': 'knowledge', 't_type': 'knowledge', 'r_type': '因果关系'}