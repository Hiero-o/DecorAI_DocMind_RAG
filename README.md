# CKP02 — DocMind RAG | DecorAI

**DecorAI** é um assistente conversacional voltado à decoração residencial, ergonomia, iluminação, cores e acabamentos. O projeto implementa um pipeline de Retrieval-Augmented Generation (RAG): consulta documentos PDF, recupera trechos relacionados à pergunta e usa o modelo de linguagem para formular uma resposta fundamentada, com citações da fonte e do chunk recuperado.

> **Estado da implementação:** o pipeline de recuperação e geração está implementado e inclui comparação entre chunks de 512 e 1024 caracteres, com e sem reranking. Há uma limitação técnica documentada: os embeddings `nomic-embed-text` são calculados localmente no ambiente Colab, enquanto a geração e o LLM avaliador usam `gemma4:cloud` no Ollama Cloud. Se a rubrica exigir embeddings obrigatoriamente via Ollama Cloud, esse ponto precisa ser resolvido ou aprovado como desvio.

## Integrantes

Júlia Lemos Souza - RM 569089
Erick Yu Xiang Li - RM 569305
Erick Banhos de Castro - RM 572131
Victor Henrique Nogueira Bezerra Azevedo de Souza - RM 570021
Carlos Henrique - RM 573334
Gustavo Araújo Ramos da Silva - RM 574156


## Objetivos

- Construir um pipeline RAG com as etapas `load → split → embed → store → retrieve → generate`.
- Responder perguntas usando o conteúdo recuperado dos documentos, evitando inventar informações ausentes.
- Citar o arquivo, a página e o identificador do chunk utilizado como origem da informação.
- Comparar dois tamanhos de chunk (512 e 1024 caracteres), com e sem reranking.
- Avaliar as respostas com as métricas `Faithfulness` e `AnswerRelevancy` do RAGAS.

## Tecnologias utilizadas

- Python e Google Colab
- PyMuPDF para leitura dos PDFs
- LangChain e LCEL para composição do pipeline
- `RecursiveCharacterTextSplitter` para divisão dos documentos
- Ollama com `nomic-embed-text` para embeddings locais no Colab
- ChromaDB para armazenamento vetorial
- Ollama Cloud com `gemma4:cloud` para geração e avaliação
- Sentence Transformers com `BAAI/bge-reranker-v2-m3` para reranking
- RAGAS para avaliação das respostas
- Pandas para organizar e exportar os resultados

## Documentos utilizados

Os PDFs abaixo compõem a base de conhecimento inicial. Ao reutilizar ou redistribuir os documentos, respeite as condições e os direitos dos respectivos autores e sites.

| Arquivo | Tema / origem informada | Link de referência |
|---|---|---|
| `Acabamentos.pdf` | Catálogo de acabamentos Century | [Catálogo de Acabamentos Century](https://meucentury.com/wp-content/uploads/2025/10/Catalogo-de-Acabamentos-Century.pdf) |
| `ergonomia.pdf` | Dissertação sobre ergonomia residencial — USP | [Documento da USP](https://teses.usp.br/teses/disponiveis/16/16132/tde-19092007-134320/publico/Dissertacao.pdf) |
| `Estilos_em_Decoracao_ABNT.pdf` | Material sobre estilos de decoração — SlideShare | [Apresentação no SlideShare](https://pt.slideshare.net/slideshow/estilos-em-decorao/63532317) |
| `ManualdeiluminacaoeficienteProcel.pdf` | Manual de iluminação eficiente — Procel / gov.br | [Manual de Iluminação Eficiente](https://www.gov.br/mme/pt-br/assuntos/ee/publicacoes-e-estudos/ManualdeiluminacaoeficienteProcel.pdf) |
| `psicologia-das-cores.pdf` | E-book sobre psicologia das cores — Printi | [E-book Psicologia das Cores](https://www.printi.com.br/files/Blog/E-book/e-book-psicologia-das-cores.pdf) |

## Como executar no Google Colab

### 1. Abrir o notebook e disponibilizar os arquivos

1. Abra `CKP02_DecorAI_limpo.ipynb` no Google Colab.
2. Envie os cinco PDFs para a pasta `/content/docs`. O ZIP deste projeto contém uma pasta `docs/` com os arquivos usados na execução inicial.
3. Se for extrair o ZIP no Colab, extraia-o diretamente em `/content` para que o caminho `/content/docs` seja criado. Se os PDFs forem enviados manualmente, crie `/content/docs` e coloque os cinco arquivos nessa pasta.

O notebook verifica se há pelo menos cinco PDFs nessa pasta e interrompe a execução se ela não existir ou estiver incompleta.

### 2. Configurar o segredo do Ollama Cloud

O notebook espera uma chave de API guardada nos Secrets do Colab:

1. Abra a área **Secrets** (ícone de chave) no Colab.
2. Crie um segredo chamado `OLLAMA_API_KEY`.
3. Cole nele sua chave do Ollama Cloud e permita que o notebook a acesse.

Não escreva a chave diretamente no notebook, no README ou em arquivos versionados.

### 3. Executar as células

Execute as células na ordem, de cima para baixo. O notebook instala as dependências, configura o ambiente, carrega os documentos, cria os chunks, inicializa o Ollama local, calcula embeddings, armazena os vetores, configura o modelo de geração e executa as avaliações.

A inicialização pode levar alguns minutos: o ambiente baixa bibliotecas e o modelo de embeddings; o reranker também pode precisar baixar seus arquivos. Para o reranking, o notebook usa GPU se estiver disponível e recorre à CPU caso contrário.

## Pipeline RAG

1. **Load:** PyMuPDF extrai texto página a página dos PDFs. Os documentos recebem metadados como arquivo de origem e número da página.
2. **Split:** `RecursiveCharacterTextSplitter` divide o texto usando os separadores `[`\\n\\n`, `\\n`, `. `, espaço, vazio`]`.
   - Configuração 512: `chunk_size=512`, `chunk_overlap=64`.
   - Configuração 1024: `chunk_size=1024`, `chunk_overlap=128`.
3. **Embed:** `nomic-embed-text` gera os vetores pelo Ollama executado localmente no Colab.
4. **Store:** os vetores são armazenados no ChromaDB, em `/content/chroma_db`, nas coleções `decorai_chunks_512` e `decorai_chunks_1024`.
5. **Retrieve:** a busca por similaridade recupera cinco trechos. Nas configurações com reranking, são recuperados dez candidatos e o CrossEncoder reordena os resultados para selecionar os cinco primeiros.
6. **Generate:** `gemma4:cloud`, com temperatura zero, recebe a pergunta e o contexto recuperado e gera a resposta em português.
7. **Citações:** o prompt exige referências com arquivo, página e identificador do chunk. O identificador combina a origem, a página, o índice inicial e o tamanho do chunk.

## Experimentos e avaliação

O notebook compara quatro configurações:

| Configuração | Tamanho do chunk | Reranking |
|---|---:|---|
| `512_sem_reranking` | 512 caracteres | Não |
| `1024_sem_reranking` | 1024 caracteres | Não |
| `512_com_reranking` | 512 caracteres | Sim |
| `1024_com_reranking` | 1024 caracteres | Sim |

A avaliação utiliza cinco perguntas sobre iluminação, cores, ergonomia, escolha de iluminação e disponibilidade de valores exatos de iluminância. São avaliadas 20 respostas no total (cinco perguntas em cada uma das quatro configurações).

- **Faithfulness:** estima se as afirmações da resposta são sustentadas pelo contexto recuperado.
- **AnswerRelevancy:** estima a relevância da resposta em relação à pergunta.

A meta definida no experimento para `Faithfulness` é **0,7**. As métricas são indicadores de avaliação, não uma garantia de correção factual. Uma resposta que informa não ter dados suficientes pode receber baixa relevância; esses resultados devem ser interpretados e discutidos, não removidos apenas para aumentar a média.

O notebook inclui testes comparativos de recuperação, inspeção da ordem do reranking e validação dos IDs de chunk citados. A validação deve ser interpretada como verificação dos exemplos testados, não como prova automática de que toda citação em qualquer pergunta será correta.

## Arquivos de resultados

Ao concluir a execução, a última parte do notebook exporta os resultados para `/content`:

- `resumo_metricas_ragas.csv`: médias e estatísticas das métricas por configuração.
- `avaliacoes_ragas.csv`: métricas individuais por pergunta e configuração.
- `respostas_avaliadas.json`: respostas, contextos recuperados, referências e metadados de origem.

Baixe esses arquivos do Colab após a execução, se precisar entregá-los separadamente. Para um relatório final, utilize os resultados da execução completa feita com a versão atual do prompt e das citações.

## Como adicionar documentos à base

1. Coloque novos PDFs dentro de `/content/docs` antes de executar a célula de carregamento.
2. Execute novamente as células de carregamento, divisão, embeddings e armazenamento.
3. Confira a contagem de documentos e chunks antes de executar a recuperação e a geração.
4. Se um PDF substituir outro com o mesmo nome ou se a base mudar substancialmente, use uma sessão limpa ou recrie as coleções ChromaDB para evitar que permaneçam vetores antigos que já não correspondem aos documentos atuais.

A extração do notebook utiliza o texto extraível dos PDFs. Documentos digitalizados como imagens podem exigir OCR, que não está implementado como etapa automática neste pipeline.

## Limitações e observações técnicas

- **Embeddings locais:** atualmente, `nomic-embed-text` é executado localmente no Colab via Ollama. Isso diverge de um requisito que exija especificamente o uso de embeddings via Ollama Cloud. O endpoint Cloud apresentou problemas de autenticação na experimentação e essa diferença não deve ser omitida na entrega.
- **Dependência de serviços externos:** a geração e a avaliação dependem de acesso ao Ollama Cloud e de uma chave de API válida.
- **Sessão Colab:** `/content/chroma_db` é armazenamento local da sessão; salve os artefatos necessários antes de encerrar ou reiniciar o ambiente.
- **RAGAS:** os resultados podem variar conforme o modelo avaliador, o endpoint, a versão das bibliotecas e a resposta gerada. Use as mesmas condições para comparar configurações.
- **Citações:** o modelo é instruído a citar os chunks recuperados. A saída deve ser inspecionada; o prompt, sozinho, não garante que todas as citações estarão corretas.
- **Fontes:** a base combina materiais de diferentes origens. A origem e a qualidade de cada documento devem ser consideradas ao interpretar as respostas.

## Licença e uso dos documentos

Este repositório é uma entrega acadêmica. Os PDFs de terceiros permanecem sujeitos aos direitos e termos definidos por seus autores e distribuidores. Verifique as condições de uso antes de redistribuir publicamente os documentos ou o conjunto completo.
