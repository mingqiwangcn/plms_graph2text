class SqlQuery:
    sel_op = 'SELECT'
    agg_ops = ['', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
    cond_ops = ['=', '>', '<', 'OP']
    where_op = 'WHERE'
    and_op = 'AND' 
     
    @staticmethod
    def get_src_tag(sql_word):
        if sql_word == '=':
            word = 'EQ'
        elif sql_word == '>':
            word = 'GT'
        elif sql_word == '<':
            word = 'LT'
        elif sql_word == 'OP':
            raise ValueError('[%s] not supported' % sql_word)
        else:
            word = sql_word 
             
        chrs = [a.upper() for a in word]
        tag = '[' + '-'.join(chrs) + ']'
        return tag

