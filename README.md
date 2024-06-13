# mDetoxifier Multilingual unsupervised text detoxifier
# Description
The mDetoxifier Multilingual unsupervised text detoxifier is a method to detoxify toxic text. The method is applicable for any language.An user can use this method to convert a toxic text into neutral one while preserving its original meaning..  

### Keywords
text-detoxification, toxicity,mask-prediction, sentence-similarity, sequence-to-sequence models

## Relevant research questions that could be adressed with the help of this method 

1.  Presence of offensive language in tweets in social media and how to detoxify them (S. Poria, E. Cambria, D. Hazarika, P. Vij, A deeper look into sarcastic tweets using deep convolutional neural networks, arXiv preprint arXiv:1610.08815 (2016))
2. Detecting and neutralising hate speech in various social media platforms (P. Liu, J. Guberman, L. Hemphill, A. Culotta, Forecasting the presence and intensity of hostility on instagram using linguistic and social features, in: Proceedings of the International AAAI Conference on Web and Social Media, volume 12, 2018.)
3. Toxicity mitigation for low-resource languages (P. Liu, J. Guberman, L. Hemphill, A. Culotta, Forecasting the presence and intensity of hostility on instagram using linguistic and social features, in: Proceedings of the International AAAI Conference on Web and Social Media, volume 12, 2018)


### Social Science Usecase

John is a researcher studying about online discourse. He wants acess to fake news within a definite time period about the US Presidential Elections. He visits the MH portal to find this method that helps him to generate a repository of fact-checked claims. He uses the search box on the top of the interface and types in Claims or Fake News. The search functionality of the MH shows him a list or related methods and tutorials that provides John with methods that can help him generate this huge collection of claims which he can then querry and find all relevant claims regarding Presidential Elections and reuse for his study.

Mary is a researcher who wants to investigate the impact of Gun laws on the society. She has a huge collection of claims, from different websites but wants to have them all at one place and search those pertaining to gun laws over a particular time period. She uses the search box to find methods related to claims or fact-checks.The search functionality of the MH shows her a list of related methods and tutorials related to claims. She then uses Verified Claims Wizard that generates a huge repository of claims out of it. She then searches this repository regarding all claims related to the Gun laws and it brings her a list of all relevant claims, be it true, false, mixed or others which she can reuse for her study.


Lily is a researcher who wants to study the evolution of false claims related to Covid or coronavirus. She collects claims from a number of fact-checking websites but does not have an easy way to pick only those that are false and also related to Covid. She uses the search box in MH to find methods related to fact-checking.The search functionality of the MH shows her a list or related methods and tutorials related to Fact-Checking that can help her generate a fact checked claims repository out of it.She generates the repository using Verified Claims Wizard and runs a search querry to find all false claims related to covid or coronavirus in a very short time. 


### Repository Structure


mdetox.py - The main file to run the project

### Environment Setup
This program requires Python 3.x to run.



  

### How to Use
Call the method in the following way
masked_sentences = mask_similar_words(hi, hindi_sentence,selected_tokens)
where hi are the toxic words in hindi language and hindi_sentences are a list of toxic sentences in hindi. The same method can be applied to any languages where the the toxic words and sentences can be replaced with the desired language



### Digital Behavioral data

### Sample Input 
A list of toxic sentences. Can be anything, for example a sample list of toxic test for multiple languages can be found here https://huggingface.co/datasets/textdetox/multilingual_paradetox_test

### Sample Output
A list of detoxified sentences along with their corrosponding toxic ones and the detected language
![](results_languages.PNG)

### mdetox pipeline

1. Language Detection Module
The first step a toxic text passes through is a language detection module. We used the Python langdetect5 library for this purpose. 
2. Toxic Words Identification and Masking 
To identify toxic words in the sentences, we adopted a combination of hashing-based techniques and log-odds ratio. As a starting point, we utilized the list of toxic lexicons.We employed a hashing-based sequence-matching mechanism7 to identify words similar to these lexicons beyond a certain threshold. These identified toxic words were then removed from the sentences and replaced with masks.
3. Mask Placement with Linguistic Patterns 
Languages follow certain grammatical paradigms or linguistic rules that aid in constructing sentences. By observing these rules, we were able to better process the masks in sentences. An example is showed below
![](linguistic_patterns.PNG)
4. Mask Prediction
Following the process of identifying and masking toxic words, and implementing linguistic rules, we were left with sentences containing masked toxic words. To handle these, we used the XLM-RoBERTa large model .Using this model, we predicted the top three probable replacements for each mask and generated sentences accordingly
5. Sentence Similarity
From our resultant sentences we chose the one that had the lowest score indicating resultant sentence closest to the input toxic sentence as our selected output sentence



### Limitation
The method needs a list of toxic lexicons(curse words in specific languages). A list of toxic lexicons for 9 languages are provided here. User can add/edit as per need and will https://huggingface.co/datasets/textdetox/multilingual_toxic_lexicon



## Contact
Susmita.Gangopadhyay@gesis.org

## Publication 
1. HybridDetox: Combining Supervised and Unsupervised Methods for Effective Multilingual Text Detoxification (Susmita Gangopadhyay, M.Taimoor Khan and Hajira Jabeen) In review for PAN CLEF Multilingual Text Detoxification Challenge

  









