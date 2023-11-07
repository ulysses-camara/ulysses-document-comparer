import os


def table_name(name):
    return "".join([c if c.isalnum() else "_" for c in name])


tb_corpus = table_name(os.getenv('TB_CORPUS', default='corpus'))
tb_solicitacoes = table_name(os.getenv('TB_SOLICITACOES', default='solicitacoes'))
tb_feedback = table_name(os.getenv('TB_FEEDBACK', default='feedback'))
