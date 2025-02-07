class Stemmer:
    def __init__(self):
        self.vowels = {'a', 'e', 'i', 'o', 'u', 'y'}
        
    def _is_vowel(self, char):
        return char.lower() in self.vowels
        
    def _contains_vowel(self, word):
        return any(self._is_vowel(char) for char in word)
        
    def _measure(self, stem):
        """Count syllable-like units in the stem"""
        v = False
        count = 0
        for i in range(len(stem)):
            is_v = self._is_vowel(stem[i])
            if v != is_v:
                if not v:
                    count += 1
                v = is_v
        return count
        
    def _replace_suffix(self, word, suffix, replacement):
        if word.endswith(suffix):
            return word[:-len(suffix)] + replacement
        return word
        
    def _apply_rule_list(self, word, rules):
        """Apply the first matching rule"""
        for suffix, replacement, condition in rules:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if condition is None or condition(stem):
                    return stem + replacement
        return word

    def stem(self, word, to_lowercase=True):
        """Main stemming function"""
        if to_lowercase:
            word = word.lower()
            
        if len(word) <= 2:
            return word
            
        # Apply the stemming steps
        word = self._step1a(word)
        word = self._step1b(word)
        return word

    def _step1a(self, word):
        """Handle plurals and -ed/-ing endings"""
        rules = [
            ("sses", "ss", None),
            ("ies", "i", None),
            ("ss", "ss", None),
            ("s", "", None)
        ]
        return self._apply_rule_list(word, rules)

    def _step1b(self, word):
        """Handle -ed and -ing"""
        if word.endswith("eed"):
            stem = self._replace_suffix(word, "eed", "")
            if self._measure(stem) > 0:
                return stem + "ee"
            return word
            
        for suffix in ["ed", "ing"]:
            if word.endswith(suffix):
                stem = self._replace_suffix(word, suffix, "")
                if self._contains_vowel(stem):
                    return stem
        return word 