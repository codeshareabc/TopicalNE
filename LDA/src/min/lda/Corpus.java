package min.lda;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;

import org.apache.lucene.analysis.PorterStemFilter;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

import min.util.URLs;

public class Corpus
{
	List<int[]> documentList;
	Map<Integer, List<int[]>> doc2senetnceLists;
	Vocabulary vocabulary; 
	Map<Integer, Integer> serviceDocLocalId2CorpudId; //mapping the doc local ids to the corpus ids
	
	public Corpus()
	{
		documentList = new ArrayList<int[]>();
		doc2senetnceLists = new HashMap<Integer, List<int[]>>();
		vocabulary = new Vocabulary();
		serviceDocLocalId2CorpudId = new HashMap<Integer, Integer>();
	}
	
	// Load the corpus
	public void load(String corpusFile, int stemmer)
	{
		try
		{
			File sourceFile = new File(corpusFile);
			BufferedReader br = new BufferedReader(new FileReader(sourceFile));
			String originaLine = "";
			List<String> wordList = null;
			List<String> params = null;
			while((originaLine = br.readLine()) != null)
			{
				if(stemmer == 1)
				{
					params = tokenization(originaLine);
				}
				else
				{
					for(String s : originaLine.split(" "))
					{
						params.add(s);
					}
				}
				wordList = new ArrayList<String>();
				for(String word : params)
				{
					wordList.add(word);
				}
				addDocument(wordList);
			}
			br.close();
		} catch (Exception e)
		{
			e.printStackTrace();
		}
	}
	
	// return all documents in the corpus
	public int[][] getDocuments()
	{
		return toArray();
	}
	// add the description
    public Map<String, Integer> addDocument(List<String> document)
    {
    	Map<String, Integer> wordIds = new HashMap<String, Integer>();
        int[] doc = new int[document.size()];
        int i = 0;
        for (String word : document)
        {
            doc[i++] = vocabulary.getId(word, true);
            wordIds.put(word, doc[i-1]);
        }
        documentList.add(doc);
        return wordIds;
    }
    
    // convert documentList to array
    public int[][] toArray()
    {
    	int[][] docs = new int[this.documentList.size()][];
    	for(int i = 0; i < documentList.size(); i++)
    	{
    		docs[i] = documentList.get(i);
    	}
    	return docs;
    }

	// save the vocabulary, document and sentences ids
    public void saveFiles(String file)
    {
    	try
		{
    		// save the vocabulary
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(this.vocabulary.toString());
			bw.flush();
			bw.close();
		} catch (IOException e)
		{
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    }
    // return the vocabulary
    public Vocabulary getVocabulary()
    {
    	return vocabulary;
    }
    
    // return the vocabulary size
    public int getVocabularySize()
    {
    	return vocabulary.size();
    }
    
	public static void main(String[] args) throws Exception
	{
		Corpus corpus = new Corpus();
		corpus.load(URLs.apisFileToken, 0);
		corpus.saveFiles("");
	}
	// Text tokenization, given a text, return a tokenized list
	public List<String> tokenization(String text)
	{
		text = text.replaceAll("'", "").replaceAll("\\.", " ").replaceAll("\\:", " ").replaceAll("\\_", " ");
		
		StandardAnalyzer standarAnalyzer = new StandardAnalyzer(Version.LUCENE_35);
		TokenStream stream = new PorterStemFilter(standarAnalyzer.tokenStream("description", new StringReader(text)));
		List<String> tokenList = new ArrayList<String>();
		try
		{
			CharTermAttribute ta = null;
			String token = "";
			while (stream.incrementToken())
			{
				ta = stream.getAttribute(CharTermAttribute.class);
				token = ta.toString();
				tokenList.add(token.trim());
			}
			stream.close();
			standarAnalyzer.close();
		} catch (IOException e)
		{
			e.printStackTrace();
		}
		return tokenList;
	}
}
