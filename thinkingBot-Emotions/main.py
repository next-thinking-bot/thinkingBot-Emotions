from flask import Flask, jsonify, request

from transformers import pipeline, AutoModelForSequenceClassification, BertJapaneseTokenizer
 
# 感情分析の実行
model = AutoModelForSequenceClassification.from_pretrained('daigo/bert-base-japanese-sentiment') 
tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
nlp = pipeline("sentiment-analysis",model=model,tokenizer=tokenizer)

app = Flask(__name__)

app.config['JSON_AS_ASCII'] = False

@app.route('/api/v1/analyse')
def analyse():
  if request.args.get('text'):
    return jsonify(nlp(request.args.get('text')))
  else:
    return jsonify([]), 400
 
if __name__ == "__main__":
  app.run(host='127.0.0.1', port=8888, debug=True)
  