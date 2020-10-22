#importar bibliotecas e recursos
from flask import Flask
from flask_restful import Resource, Api, reqparse
from flask import jsonify, make_response
import os
import pickle
import unidecode
from string import punctuation
import nltk
from nltk import tokenize
nltk.download('rslp')
nltk.download('stopwords')
stemer = nltk.RSLPStemmer()

#Configuração da API
app = Flask(__name__)
api = Api(app)

class Analise(Resource):

    def get(self):
        # parse request arguments
        parser = reqparse.RequestParser()
        parser.add_argument('frase', required=True)
        args = parser.parse_args()
        analise = testa_frase(args['frase'])
        resposta = { "tipo": "Analise de sentimento", "Variavel": "Comportamento Violento", "valor": analise[0][1]}
        return make_response(jsonify(resposta), 200)
    
api.add_resource(Analise, '/analise', methods=['GET'])

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

#Tratamento da frase
token_espaco = tokenize.WhitespaceTokenizer()

token_pontuacao = tokenize.WordPunctTokenizer()

palavras_irrelevantes = nltk.corpus.stopwords.words("portuguese")

pontuacao = list()
for ponto in punctuation:
  pontuacao.append(ponto)

pontuacao_stopwords = pontuacao + palavras_irrelevantes

def trata_frase(frase):
  frase_tratada = list()
  nova_frase = list()
  frase = frase.lower()
  palavras_texto = token_espaco.tokenize(frase)
  for palavra in palavras_texto:
    if '@' not in palavra:
      if palavra not in pontuacao_stopwords:      
        nova_frase.append(stemer.stem(palavra))
  frase_tratada.append(' '.join(nova_frase))
  frase_tratada = [unidecode.unidecode(frase) for frase in frase_tratada]
  
  return frase_tratada

#importar o modelo
sgd = pickle.load(open("modelo.sav", 'rb'))
tfidf = pickle.load(open("vectorizer.pickle", 'rb'))

#Realizar a previsão
def testa_frase(frase):
  return(sgd.predict_proba(tfidf.transform(trata_frase(frase))))


