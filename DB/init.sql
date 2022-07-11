CREATE TABLE arvore_proposicoes (
   code INTEGER NOT NULL,
   numero_sequencia INTEGER,
   cod_proposicao INTEGER,
   cod_proposicao_referenciada INTEGER,
   tipo_referencia VARCHAR
);

 CREATE TABLE proposicoes_legislativas (
   code INTEGER NOT NULL,
   sig_tipo VARCHAR,
   name VARCHAR,
   txt_ementa VARCHAR,
   em_tramitacao VARCHAR,
   situacao VARCHAR,
   text VARCHAR,
   text_preprocessed VARCHAR
);

 CREATE TABLE feedback (
   code SERIAL,
   query VARCHAR,
   user_feedback VARCHAR,
   extra_results VARCHAR,
   date_created TIMESTAMP,
   user_id VARCHAR
);

CREATE TABLE consultas_legislativas (
   code INTEGER NOT NULL,
   sig_tipo VARCHAR,
   name VARCHAR,
   text VARCHAR,
   text_preprocessed VARCHAR,
   ficticia BOOLEAN
);

\COPY corpus FROM '/var/lib/postgresql/corpus.csv' CSV HEADER;
\COPY arvore_proposicoes FROM '/var/lib/postgresql/arvore-proposicoes.csv' CSV HEADER;
\COPY solicitacoes FROM '/var/lib/postgresql/solicitacoes.csv' CSV HEADER;
