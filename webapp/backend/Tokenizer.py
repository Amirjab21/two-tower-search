import collections
from Stemmer import Stemmer
import json

DEFAULT_PUNC_MAP = {
    '.': ' ~DOT',
    ',': ' ~COMMA',
    '!': ' ~EXCLAMATION',
    '?': ' ~QUESTION',
    ':': ' ~COLON',
    ';': ' ~SEMICOLON',
}

class Tokenizer():
    def __init__(self, stemmer = Stemmer(), punc_map = DEFAULT_PUNC_MAP):
        self.STOP_WORDS = ['and', 'if', 'the', 'is', 'in', 'to', 'of', 'at', 'by', 'with', 'as', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'as', 'until', 'while', 'as', 'until', 'while']
        self.stemmer = stemmer
        self.punc_map = punc_map

    def tokenize_word(self, word):

        if word == "nan" or word.strip() == "":
            return "<nan>"  # Handle missing or NaN values explicitly

        # Lowercase the word
        word_token = word.lower()
        word_token = self.stemmer.stem(word_token)

        return word_token
    

    def replace_punctuation_with_token(
        self, text: str, pmap: dict[str, str]
    ) -> str:
        for punc, token in pmap.items():
            text = text.replace(punc, token)
        return text.strip()


    def tokenize(self, text):
        text = self.replace_punctuation_with_token(text, self.punc_map)
        words = text.split()
        words = [self.tokenize_word(word) for word in words]
        return words
    
    def build_vocabulary(self, text, fileout_path):
        # print(text)/
        text = self.replace_punctuation_with_token(text, self.punc_map)
        tokenized_text = self.tokenize(text)
        # print(tokenized_text[:10])
        flattened_tokens = [token for sublist in tokenized_text for token in (sublist if isinstance(sublist, list) else [sublist])]
        # print(flattened_tokens[:10])
        stats = collections.Counter(flattened_tokens)

        top_30 = stats.most_common(30)
        top_30_words = set(word for word, _ in top_30)
        
        # Filter words: remove top 30 and words with freq < 20
        filtered_words = {
            word: freq for word, freq in stats.items() 
            if freq >= 20 and word
        }

        # print(filtered_words)
        
        # Create dictionary of excluded words
        excluded_words = {
            word: freq for word, freq in stats.items()
            if freq < 20 or word
        }
        
        # Save filtered vocabulary
        with open(fileout_path, 'w') as f:
            json.dump(filtered_words, f, indent=4)
            
        # Save excluded words
        excluded_path = fileout_path.rsplit('.', 1)[0] + '_excluded.json'
        with open(excluded_path, 'w') as f:
            json.dump(excluded_words, f, indent=4)
        
        # Print statistics
        # print(f"Total number of unique words: {len(stats)}")
        # print("\nTop 30 most frequent words:")
        # for word, freq in top_30:
        #     print(f"{word}: {freq}")
        # print(f"\nNumber of words in filtered vocabulary: {len(filtered_words)}")
        # print(f"Number of excluded words: {len(excluded_words)}")

    def get_lookup_table(self, vocab_file):
        with open(vocab_file, 'r') as f:
            vocab = json.load(f)
        # Add special tokens at the beginning
        special_tokens = ['<pad>', '<unknown>']
        word_to_id = {word: idx for idx, word in enumerate(special_tokens)}
        # Continue enumeration from len(special_tokens)
        word_to_id.update({word: idx + len(special_tokens) for idx, word in enumerate(vocab)})
        id_to_word = {idx: word for word, idx in word_to_id.items()}
        for i, (word, id) in enumerate(list(word_to_id.items())[:5]):
            print(f"{word}: {id}")
            print(f"{id_to_word[id]} === {word}")
        return word_to_id, id_to_word
        
    