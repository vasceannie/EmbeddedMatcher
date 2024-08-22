from nltk.corpus import wordnet

def get_synonyms(word):
    synonyms = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name())
    return synonyms

def synonym_match(source_category, target_categories, threshold=0.5):
    source_tokens = source_category.split()
    source_synonyms = set()
    for token in source_tokens:
        source_synonyms.update(get_synonyms(token))
    
    matches = []
    for index, target_category in target_categories.iterrows():
        target_tokens = target_category['processed_category'].split()
        target_synonyms = set()
        for token in target_tokens:
            target_synonyms.update(get_synonyms(token))
        
        similarity = len(source_synonyms.intersection(target_synonyms)) / len(source_synonyms.union(target_synonyms))
        if similarity > threshold:
            matches.append((target_category['category'], similarity))
    
    return matches

def apply_synonym_matching(source_df, target_df):
    source_df['synonym_matches'] = source_df['processed_category'].apply(lambda x: synonym_match(x, target_df))
    return source_df