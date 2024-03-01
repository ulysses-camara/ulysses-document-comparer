CREATE TABLE arvore_proposicoes (
   numero_sequencia INTEGER,
   nivel INTEGER,
   cod_proposicao INTEGER,
   cod_proposicao_referenciada INTEGER,
   cod_proposicao_raiz INTEGER,
   tipo_referencia VARCHAR
);

 CREATE TABLE corpus (
   code INTEGER NOT NULL,
   name VARCHAR,
   txt_ementa VARCHAR,
   text VARCHAR,
   text_preprocessed VARCHAR
);

 CREATE TABLE feedback (
   id SERIAL,
   query VARCHAR,
   user_feedback VARCHAR,
   extra_results VARCHAR,
   date_created TIMESTAMP,
   user_id VARCHAR
);

CREATE TABLE solicitacoes (
   code INTEGER NOT NULL,
   name VARCHAR,
   text VARCHAR,
   text_preprocessed VARCHAR,
   ficticia BOOLEAN
);

\COPY corpus FROM '/var/lib/postgresql/corpus.csv' CSV HEADER;
\COPY arvore_proposicoes FROM '/var/lib/postgresql/arvore-proposicoes.csv' CSV HEADER;
\COPY solicitacoes FROM '/var/lib/postgresql/solicitacoes.csv' CSV HEADER;
