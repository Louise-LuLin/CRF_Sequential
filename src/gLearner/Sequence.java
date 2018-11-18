package gLearner;

import java.util.ArrayList;

public class Sequence {
	String m_content;
	
	ArrayList<Token> m_tokenList;
	
	public Sequence(String content) {
		m_content = content;
		
		m_tokenList = new ArrayList<Token>();
	}
	
	void addToken(String content, int index) {
		m_tokenList.add(new Token(content, index));
	}
	
	void assignLabel(int tokenIdx, int label) {
		if (tokenIdx < m_tokenList.size()) 
			m_tokenList.get(tokenIdx).m_label = label;
		else 
			System.err.format("Token index %d out of sequence range %d!", tokenIdx, m_tokenList.size());
	}
	
	int[] getLabels() {
		int[] labels = new int[m_tokenList.size()];
		
		for(int i=0; i<labels.length; i++)
			labels[i] = m_tokenList.get(i).m_label;
		
		return labels;
	}
	
	int[] getTokens() {
		int[] tokens = new int[m_tokenList.size()];
		
		for(int i=0; i<tokens.length; i++)
			tokens[i] = m_tokenList.get(i).m_index;
		
		return tokens;
	}
	
	int getLength() {
		return m_tokenList.size();
	}
}
