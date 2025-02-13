import nltk
import re
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

def group_sentences(text, max_length=400):
    """
    Provided splitting logic, with minor edits to ensure it runs in this script.
    Splits text into chunks that do not exceed max_length. 
    """
    print(f"\nInitial text: {text}")
    sentences = sent_tokenize(text)
    print(f"Initial sentences: {sentences}")
    print(f"Number of initial sentences: {len(sentences)}")

    # Debug: print the length of the longest sentence
    print(f"Longest sentence: {max([len(sent) for sent in sentences]) if sentences else 0}")

    split_iteration = 0
    previous_max_length = 0

    # While biggest sentence is bigger than max_length, try splitting by periods, then by commas
    while sentences and max([len(sent) for sent in sentences]) > max_length:
        split_iteration += 1
        if split_iteration > 10:
            break
        max_sent = max(sentences, key=len)
        max_idx = sentences.index(max_sent)
        
        # Add check for previous split attempts
        if split_iteration > 1 and len(max_sent) == previous_max_length:
            print("Can't split further, breaking loop")
            break
        previous_max_length = len(max_sent)
        
        print(f"\nSplitting sentence: {max_sent}")
        print(f"Length of sentence being split: {len(max_sent)}")

        sentences_before = sentences[:max_idx]
        sentences_after = sentences[max_idx+1:]
        
        new_sentences = max_sent.split(".")
        new_sentences = [sent.strip() for sent in new_sentences if sent.strip() != ""]
        print(f"After period split: {new_sentences}")
        
        # check if a split sentence is still too big
        if new_sentences and max([len(sent) for sent in new_sentences]) > max_length:
            biggest_new_sent = max(new_sentences, key=len)
            print(f"\nStill too big, splitting by comma: {biggest_new_sent}")
            
            # Add check if comma exists before attempting split
            if ',' not in biggest_new_sent:
                print("No commas to split, can't reduce further")
                break  # Exit comma splitting attempt
                
            bn_idx = new_sentences.index(biggest_new_sent)
            new_senteces_before = new_sentences[:bn_idx]
            new_senteces_after = new_sentences[bn_idx+1:]

            # Split on commas but keep them with the preceding text
            new_sentence_parts = re.split(r'(?<=,)', biggest_new_sent)
            new_sentence_parts = [sent.strip() for sent in new_sentence_parts if sent.strip() != ""]
            print(f"After comma split: {new_sentence_parts}")
            new_sentences = new_senteces_before + new_sentence_parts + new_senteces_after
        
        sentences = sentences_before + new_sentences + sentences_after
        print(f"Current sentences after splitting: {sentences}")

    # Debug: print the length of the longest sentence after splitting
    if sentences:
        print(f"\nLongest sentence after split: {max([len(sent) for sent in sentences])}")
        print(f"All sentences before merging: {sentences}")

    # Merge sentences until we can't merge further without exceeding max_length
    merge_iteration = 0
    while True:
        if len(sentences) <= 1:
            print("\nOnly one sentence left, stopping merge")
            break

        merge_iteration += 1
        print(f"\nMerge iteration {merge_iteration}:")

        if merge_iteration > 10:
            break
        
        # Find the shortest sentence
        min_index = min(range(len(sentences)), key=lambda i: len(sentences[i]))
        min_length = len(sentences[min_index])
        print(f"Shortest sentence: '{sentences[min_index]}' at index {min_index}")

        # Determine the nearest neighbor that is shorter (or equally short)
        if min_index == 0:
            next_index = min_index + 1
        elif min_index == len(sentences) - 1:
            next_index = min_index - 1
        else:
            left_length = len(sentences[min_index - 1])
            right_length = len(sentences[min_index + 1])
            next_index = (min_index - 1) if left_length <= right_length else (min_index + 1)
        
        print(f"Considering merge with sentence at index {next_index}: '{sentences[next_index]}'")
        print(f"Combined length would be: {min_length + len(sentences[next_index]) + 1}")

        # Check if merging would exceed the maximum length
        if min_length + len(sentences[next_index]) + 1 > max_length:
            print("Merge would exceed max_length, stopping")
            break

        # Merge sentences
        if next_index > min_index:
            sentences[min_index] = f"{sentences[min_index]} {sentences[next_index]}"
            del sentences[next_index]
        else:
            sentences[next_index] = f"{sentences[next_index]} {sentences[min_index]}"
            del sentences[min_index]
        print(f"Sentences after merge: {sentences}")

    return sentences


if __name__ == "__main__":
    text = "Hi! I'm Zonos, a text-to-speech model build by Zyphra. I can speak 5 languages and you can even control my emotions. You can also find me in Kuluko, an app that lets you create fully personalized audiobook, from characters to storylines, all tailored to your preferences. Want to give it a go? Search for Kuluko on the Apple or Android app store and start crafting your own story today!"
    print(group_sentences(text, 100))