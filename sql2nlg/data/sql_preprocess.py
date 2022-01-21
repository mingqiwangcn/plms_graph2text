import json
from sql_data import SqlQuery
from tqdm import tqdm

def read_tables(table_file):
    table_dict = {}
    with open(table_file) as f:
        for line in tqdm(f):
            item = json.loads(line)
            table_id = item['id']
            table_dict[table_id] = item
    return table_dict

def process(sql_file, table_file, out_dir):
    out_src_file = out_dir + '/source.txt'
    out_tgt_file = out_dir + '/target.txt'

    f_o_src = open(out_src_file, 'w')
    f_o_tgt = open(out_tgt_file, 'w')

    table_dict = read_tables(table_file)
    with open(sql_file) as f:
        #row = -1
        for line in tqdm(f):
            #row += 1
            #print('\nrow=%d' % row)
            item = json.loads(line)
            table_id = item['table_id']
            table_info = table_dict[table_id]
            question = item['question']
            sql_info = item['sql'] 
            sql_text = get_sql_text(table_info, sql_info)
            if len(sql_info['conds']) > 1:
                print()
            
            f_o_src.write(sql_text + '\n')
            f_o_tgt.write(question + '\n')
    
    f_o_src.close()
    f_o_tgt.close()

def get_sql_text(table_info, sql_info):
    sel_col_idx = sql_info['sel']
    agg_op_idx = sql_info['agg']
    conds = sql_info['conds']
    
    col_names = table_info['header']
    sel_col_name = col_names[sel_col_idx]
    
    agg_op = SqlQuery.agg_ops[agg_op_idx]
    if agg_op != '':
        agg_op_tag = SqlQuery.get_src_tag(agg_op)
    else:
        agg_op_tag = ''

    cond_text_lst = []
    for cond_info in conds:
        col_idx, op_idx, cond_value = cond_info
        cond_col_name = col_names[col_idx]
        cond_op = SqlQuery.cond_ops[op_idx]
        cond_op_tag = SqlQuery.get_src_tag(cond_op)
        cond_text = '%s %s %s' % (cond_col_name, cond_op_tag, cond_value)
        cond_text_lst.append(cond_text)
   
    sel_op_tag = SqlQuery.get_src_tag(SqlQuery.sel_op) 
    where_op_tag = SqlQuery.get_src_tag(SqlQuery.where_op)
    and_op_tag = ' ' + SqlQuery.get_src_tag(SqlQuery.and_op) + ' '
    cond_expr = and_op_tag.join(cond_text_lst)
    if agg_op_tag != '':
        sql_text = ' '.join([sel_op_tag, agg_op_tag, sel_col_name, where_op_tag, cond_expr])
    else:
        sql_text = ' '.join([sel_op_tag, sel_col_name, where_op_tag, cond_expr])
    return sql_text

def get_files(mode):
    sql_file = '/home/cc/data/wikisql/%s.jsonl' % mode
    table_file = '/home/cc/data/wikisql/%s.tables.jsonl' % mode
    
    return sql_file, table_file

def main():
    train_sql_file, train_table_file = get_files('train')
    process(train_sql_file, train_table_file, './output') 


if __name__ == '__main__':
    main()
