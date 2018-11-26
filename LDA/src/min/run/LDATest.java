package min.run;

import java.util.List;

import min.lda.LDA;
import min.lda.Corpus;
import min.util.URLs;
import org.junit.Test;

public class LDATest
{
	@Test
	// Train the Sen-LDA model
	public void TestMain() throws Exception
	{
		int T = 35; // The number of latent topics
		double alpha = 0.1; // The prior hyperparameter of documentToTopic 
		double beta = 0.05; // The prior hyperparameter of topicToTerm
		int iters = 2000; // The iteration time
//		 1. Load corpus from disk
		Corpus corpus = new Corpus();
		corpus.load("workspace_python\\TopicalNE\\datasets\\cora\\content_token.txt", 1);
		corpus.saveFiles("workspace_python\\TopicalNE\\datasets\\cora\\vocabulary.txt");
        // 2. Create a Att-LDA sampler
		LDA lda = new LDA(corpus.getDocuments(), corpus.getVocabularySize());
        // 3. Training
		lda.gibbs(T, alpha, beta, iters);
        // 4. Save model
        String modelName = "T_" + T;
        lda.saveModel(modelName, "workspace_python\\TopicalNE\\datasets\\cora");
        
        // 5. Calculate the top-k similar terms for a given term
//		LDA attlda = new LDA();
//		String word = "vertex";
//		List<String> topWords = attlda.getTopKNeighbors(word, 500);
//		System.out.println("The source word is: " + word + "\n");
//		for(String s : topWords)
//		{
//			System.out.println(s);
//		}
	}
}